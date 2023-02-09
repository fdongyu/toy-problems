static char help[] = "Partial 2D dam break problem.\n";

#include <assert.h>
#include <math.h>
#include <petsc.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <petscts.h>
#include <petscvec.h>

PetscReal GRAVITY = 9.806;

#define Square(x) ((x) * (x))

/// Allocates a block of memory of the given type, consisting of count
/// contiguous elements and placing the allocated memory in the given result
/// pointer. Memory is zero-initialized. Returns a PetscErrorCode.
#define RDyAlloc(type, count, result) PetscCalloc1(sizeof(type) * (count), result)

/// Frees a block of memory allocated by RDyAlloc. Returns a PetscErrorCode.
#define RDyFree(memory) PetscFree(memory)

/// Fills an array of the given type and given element count with the given
/// value, performing an explicit cast for each value. Returns a 0 error code,
/// as it cannot fail under detectable conditions.
/// NOTE: Note the leading "0", which provides the return code. This trick may
/// NOTE: produce an "unused value" warning with certain compiler settings if
/// NOTE: the error code is not captured by the caller. Since we use the
/// NOTE: PetscCall(func(args)) convention, this shouldn't be an issue.
#define RDyFill(type, memory, count, value) \
  0;                                        \
  for (size_t i = 0; i < (count); ++i) {    \
    memory[i] = (type)value;                \
  }

/// Returns true iff start <= closure < end.
PetscBool IsClosureWithinBounds(PetscInt closure, PetscInt start, PetscInt end) { return (closure >= start) && (closure < end); }

/// a point in R^3
typedef struct {
  PetscReal X[3];
} RDyPoint;

/// a vector in R^3
typedef struct {
  PetscReal V[3];
} RDyVector;

/// a type indicating one of a set of supported cell types
typedef enum {
  CELL_TRI_TYPE = 0,  // tetrahedron cell for a 3D cell
  CELL_QUAD_TYPE      // hexahedron cell for a 3D cell
} RDyCellType;

/// A struct of arrays storing information about mesh cells. The ith element in
/// each array stores a property for mesh cell i.
typedef struct {
  /// local IDs of cells in local numbering
  PetscInt *ids;
  /// global IDs of cells in local numbering
  PetscInt *global_ids;
  /// natural IDs of cells in local numbering
  PetscInt *natural_ids;

  /// PETSC_TRUE iff corresponding cell is locally stored
  PetscBool *is_local;

  /// numbers of cell vertices
  PetscInt *num_vertices;
  /// numbers of cell edges
  PetscInt *num_edges;
  /// numbers of cell neigbors (themselves cells)
  PetscInt *num_neighbors;

  /// offsets of first cell vertices in vertex_ids
  PetscInt *vertex_offsets;
  /// IDs of vertices for cells
  PetscInt *vertex_ids;

  /// offsets of first cell edges in edge_ids
  PetscInt *edge_offsets;
  /// IDs of edges for cells
  PetscInt *edge_ids;

  /// offsets of first neighbors in neighbor_ids for cells
  PetscInt *neighbor_offsets;
  /// IDs of neighbors for cells
  PetscInt *neighbor_ids;

  /// cell centroids
  RDyPoint *centroids;
  /// cell areas
  PetscReal *areas;

  /// surface slope in x-dir
  PetscReal *dz_dx;
  /// surface slope in y-dir
  PetscReal *dz_dy;

} RDyCells;

/// Allocates and initializes an RDyCells struct.
/// @param [in] num_cells Number of cells
/// @param [out] cells A pointer to an RDyCells that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyCellsCreate(PetscInt num_cells, RDyCells *cells) {
  PetscFunctionBegin;

  PetscInt vertices_per_cell  = 4;
  PetscInt edges_per_cell     = 4;
  PetscInt neighbors_per_cell = 4;

  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->ids));
  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->global_ids));
  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->natural_ids));
  PetscCall(RDyFill(PetscInt, cells->global_ids, num_cells, -1));
  PetscCall(RDyFill(PetscInt, cells->natural_ids, num_cells, -1));

  PetscCall(RDyAlloc(PetscBool, num_cells, &cells->is_local));
  PetscCall(RDyFill(PetscInt, cells->is_local, num_cells, PETSC_FALSE));

  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->num_vertices));
  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->num_edges));
  PetscCall(RDyAlloc(PetscInt, num_cells, &cells->num_neighbors));
  PetscCall(RDyFill(PetscInt, cells->num_vertices, num_cells, -1));
  PetscCall(RDyFill(PetscInt, cells->num_edges, num_cells, -1));
  PetscCall(RDyFill(PetscInt, cells->num_neighbors, num_cells, -1));

  PetscCall(RDyAlloc(PetscInt, num_cells + 1, &cells->vertex_offsets));
  PetscCall(RDyAlloc(PetscInt, num_cells + 1, &cells->edge_offsets));
  PetscCall(RDyAlloc(PetscInt, num_cells + 1, &cells->neighbor_offsets));
  PetscCall(RDyFill(PetscInt, cells->vertex_offsets, num_cells + 1, -1));
  PetscCall(RDyFill(PetscInt, cells->edge_offsets, num_cells + 1, -1));
  PetscCall(RDyFill(PetscInt, cells->neighbor_offsets, num_cells + 1, -1));

  PetscCall(RDyAlloc(PetscInt, num_cells * vertices_per_cell, &cells->vertex_ids));
  PetscCall(RDyAlloc(PetscInt, num_cells * edges_per_cell, &cells->edge_ids));
  PetscCall(RDyAlloc(PetscInt, num_cells * neighbors_per_cell, &cells->neighbor_ids));
  PetscCall(RDyFill(PetscInt, cells->vertex_ids, num_cells * vertices_per_cell, -1));
  PetscCall(RDyFill(PetscInt, cells->edge_ids, num_cells * edges_per_cell, -1));
  PetscCall(RDyFill(PetscInt, cells->neighbor_ids, num_cells * neighbors_per_cell, -1));

  PetscCall(RDyAlloc(RDyPoint, num_cells, &cells->centroids));
  PetscCall(RDyAlloc(PetscReal, num_cells, &cells->areas));
  PetscCall(RDyAlloc(PetscReal, num_cells, &cells->dz_dx));
  PetscCall(RDyAlloc(PetscReal, num_cells, &cells->dz_dy));

  for (PetscInt icell = 0; icell < num_cells; icell++) {
    cells->ids[icell]           = icell;
    cells->num_vertices[icell]  = vertices_per_cell;
    cells->num_edges[icell]     = edges_per_cell;
    cells->num_neighbors[icell] = neighbors_per_cell;
  }

  for (PetscInt icell = 0; icell <= num_cells; icell++) {
    cells->vertex_offsets[icell]   = icell * vertices_per_cell;
    cells->edge_offsets[icell]     = icell * edges_per_cell;
    cells->neighbor_offsets[icell] = icell * neighbors_per_cell;
  }

  PetscFunctionReturn(0);
}

/// Creates a fully initialized RDyCells struct from a given DM.
/// @param [in] dm A DM that provides cell data
/// @param [out] cells A pointer to an RDyCells that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyCellsCreateFromDM(DM dm, RDyCells *cells) {
  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscInt eStart, eEnd;
  PetscInt vStart, vEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

  // allocate cell storage
  PetscCall(RDyCellsCreate(cEnd - cStart, cells));

  PetscInt dim;
  PetscCall(DMGetCoordinateDim(dm, &dim));

  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscInt  icell = c - cStart;
    PetscInt  gref, junkInt;
    PetscReal centroid[dim], normal[dim];
    PetscCall(DMPlexGetPointGlobal(dm, c, &gref, &junkInt));
    DMPlexComputeCellGeometryFVM(dm, c, &cells->areas[icell], &centroid[0], &normal[0]);

    for (PetscInt idim = 0; idim < dim; idim++) {
      cells->centroids[icell].X[idim] = centroid[idim];
    }

    PetscInt  pSize;
    PetscInt *p        = NULL;
    PetscInt  use_cone = PETSC_TRUE;

    cells->num_vertices[icell] = 0;
    cells->num_edges[icell]    = 0;
    if (gref >= 0) {
      cells->is_local[icell] = PETSC_TRUE;
    } else {
      cells->is_local[icell] = PETSC_FALSE;
    }

    PetscCall(DMPlexGetTransitiveClosure(dm, c, use_cone, &pSize, &p));
    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], eStart, eEnd)) {
        PetscInt offset        = cells->edge_offsets[icell];
        PetscInt index         = offset + cells->num_edges[icell];
        cells->edge_ids[index] = p[i] - eStart;
        cells->num_edges[icell]++;
      } else {
        PetscInt offset          = cells->vertex_offsets[icell];
        PetscInt index           = offset + cells->num_vertices[icell];
        cells->vertex_ids[index] = p[i] - vStart;
        cells->num_vertices[icell]++;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, use_cone, &pSize, &p));
  }

  PetscFunctionReturn(0);
}

/// Destroys an RDyCells struct, freeing its resources.
/// @param [inout] cells An RDyCells struct whose resources will be freed.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyCellsDestroy(RDyCells cells) {
  PetscFunctionBegin;

  PetscCall(RDyFree(cells.ids));
  PetscCall(RDyFree(cells.global_ids));
  PetscCall(RDyFree(cells.natural_ids));
  PetscCall(RDyFree(cells.is_local));
  PetscCall(RDyFree(cells.num_vertices));
  PetscCall(RDyFree(cells.num_edges));
  PetscCall(RDyFree(cells.num_neighbors));
  PetscCall(RDyFree(cells.vertex_offsets));
  PetscCall(RDyFree(cells.edge_offsets));
  PetscCall(RDyFree(cells.neighbor_offsets));
  PetscCall(RDyFree(cells.vertex_ids));
  PetscCall(RDyFree(cells.edge_ids));
  PetscCall(RDyFree(cells.neighbor_ids));
  PetscCall(RDyFree(cells.centroids));
  PetscCall(RDyFree(cells.areas));

  PetscFunctionReturn(0);
}

/// A struct of arrays storing information about mesh vertices. The ith element
/// in each array stores a property for vertex i.
typedef struct {
  /// local IDs of vertices in local numbering
  PetscInt *ids;
  /// global IDs of vertices in local numbering
  PetscInt *global_ids;

  /// PETSC_TRUE iff vertex is attached to a local cell
  PetscBool *is_local;

  /// numbers of cells attached to vertices
  PetscInt *num_cells;
  /// numbers of edges attached to vertices
  PetscInt *num_edges;

  /// offsets of first vertex edges in edge_ids
  PetscInt *edge_offsets;
  /// IDs of edges attached to vertices
  PetscInt *edge_ids;

  /// offsets of first vertex cells in cell_ids
  PetscInt *cell_offsets;
  /// IDs of local cells attached to vertices
  PetscInt *cell_ids;

  /// vertex positions
  RDyPoint *points;
} RDyVertices;

/// Allocates and initializes an RDyVertices struct.
/// @param [in] num_vertices Number of vertices
/// @param [out] vertices A pointer to an RDyVertices that stores data
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyVerticesCreate(PetscInt num_vertices, RDyVertices *vertices) {
  PetscFunctionBegin;

  PetscInt cells_per_vertex = 4;
  PetscInt edges_per_vertex = 4;

  PetscCall(RDyAlloc(PetscInt, num_vertices, &vertices->ids));
  PetscCall(RDyAlloc(PetscInt, num_vertices, &vertices->global_ids));
  PetscCall(RDyAlloc(PetscInt, num_vertices, &vertices->num_cells));
  PetscCall(RDyAlloc(PetscInt, num_vertices, &vertices->num_edges));
  PetscCall(RDyFill(PetscInt, vertices->global_ids, num_vertices, -1));

  PetscCall(RDyAlloc(PetscBool, num_vertices, &vertices->is_local));

  PetscCall(RDyAlloc(RDyPoint, num_vertices, &vertices->points));

  PetscCall(RDyAlloc(PetscInt, num_vertices + 1, &vertices->edge_offsets));
  PetscCall(RDyAlloc(PetscInt, num_vertices + 1, &vertices->cell_offsets));

  PetscCall(RDyAlloc(PetscInt, num_vertices * edges_per_vertex, &vertices->edge_ids));
  PetscCall(RDyAlloc(PetscInt, num_vertices * cells_per_vertex, &vertices->cell_ids));
  PetscCall(RDyFill(PetscInt, vertices->edge_ids, num_vertices * edges_per_vertex, -1));
  PetscCall(RDyFill(PetscInt, vertices->cell_ids, num_vertices * cells_per_vertex, -1));

  for (PetscInt ivertex = 0; ivertex < num_vertices; ivertex++) {
    vertices->ids[ivertex] = ivertex;
    for (PetscInt idim = 0; idim < 3; idim++) {
      vertices->points[ivertex].X[idim] = 0.0;
    }
  }

  for (PetscInt ivertex = 0; ivertex <= num_vertices; ivertex++) {
    vertices->edge_offsets[ivertex] = ivertex * edges_per_vertex;
    vertices->cell_offsets[ivertex] = ivertex * cells_per_vertex;
  }

  PetscFunctionReturn(0);
}

/// Creates a fully initialized RDyVertices struct from a given DM.
/// @param [in] dm A DM that provides vertex data
/// @param [out] vertices A pointer to an RDyVertices that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyVerticesCreateFromDM(DM dm, RDyVertices *vertices) {
  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscInt eStart, eEnd;
  PetscInt vStart, vEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

  // allocate vertex storage
  PetscCall(RDyVerticesCreate(vEnd - vStart, vertices));

  PetscSection coordSection;
  Vec          coordinates;
  DMGetCoordinateSection(dm, &coordSection);
  DMGetCoordinatesLocal(dm, &coordinates);
  PetscReal *coords;
  VecGetArray(coordinates, &coords);

  for (PetscInt v = vStart; v < vEnd; v++) {
    PetscInt  ivertex = v - vStart;
    PetscInt  pSize;
    PetscInt *p = NULL;

    PetscCall(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &pSize, &p));

    PetscInt coordOffset, dim;
    PetscCall(DMGetCoordinateDim(dm, &dim));
    PetscSectionGetOffset(coordSection, v, &coordOffset);
    for (PetscInt idim = 0; idim < dim; idim++) {
      vertices->points[ivertex].X[idim] = coords[coordOffset + idim];
    }

    vertices->num_edges[ivertex] = 0;
    vertices->num_cells[ivertex] = 0;

    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], eStart, eEnd)) {
        PetscInt offset           = vertices->edge_offsets[ivertex];
        PetscInt index            = offset + vertices->num_edges[ivertex];
        vertices->edge_ids[index] = p[i] - eStart;
        vertices->num_edges[ivertex]++;
      } else {
        PetscInt offset           = vertices->cell_offsets[ivertex];
        PetscInt index            = offset + vertices->num_cells[ivertex];
        vertices->cell_ids[index] = p[i] - cStart;
        vertices->num_cells[ivertex]++;
      }
    }

    PetscCall(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &pSize, &p));
  }

  VecRestoreArray(coordinates, &coords);

  PetscFunctionReturn(0);
}

/// Destroys an RDyVertices struct, freeing its resources.
/// @param [inout] vertices An RDyVertices struct whose resources will be freed
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyVerticesDestroy(RDyVertices vertices) {
  PetscFunctionBegin;

  PetscCall(RDyFree(vertices.ids));
  PetscCall(RDyFree(vertices.global_ids));
  PetscCall(RDyFree(vertices.is_local));
  PetscCall(RDyFree(vertices.num_cells));
  PetscCall(RDyFree(vertices.num_edges));
  PetscCall(RDyFree(vertices.cell_offsets));
  PetscCall(RDyFree(vertices.edge_offsets));
  PetscCall(RDyFree(vertices.cell_ids));
  PetscCall(RDyFree(vertices.edge_ids));
  PetscCall(RDyFree(vertices.points));

  PetscFunctionReturn(0);
}

/// A struct of arrays storing information about edges separating mesh cells.
/// The ith element in each array stores a property for edge i.
typedef struct {
  /// local IDs of edges in local numbering
  PetscInt *ids;
  /// global IDs of edges in local numbering
  PetscInt *global_ids;
  /// local IDs of internal edges
  PetscInt *internal_edge_ids;
  /// local IDs of boundary edges
  PetscInt *boundary_edge_ids;

  /// PETSC_TRUE if edge is shared by locally owned cells, OR
  /// if it is shared by a local cell c1 and non-local cell c2 such that
  /// global ID(c1) < global ID(c2).
  PetscBool *is_local;

  /// numbers of cells attached to edges
  PetscInt *num_cells;
  /// numbers of vertices attached to edges
  PetscInt *vertex_ids;

  /// offsets of first edge cells in cell_ids
  PetscInt *cell_offsets;
  /// IDs of cells attached to edges
  PetscInt *cell_ids;

  /// false if the edge is on the domain boundary
  PetscBool *is_internal;

  /// unit vector pointing out of one cell into another for each edge
  RDyVector *normals;

  /// edge centroids
  RDyPoint *centroids;

  /// edge lengths
  PetscReal *lengths;

  /// cosine of the angle between edge and y-axis
  PetscReal *cn;

  /// sine of the angle between edge and y-axis
  PetscReal *sn;

} RDyEdges;

/// Allocates and initializes an RDyEdges struct.
/// @param [in] num_edges Number of edges
/// @param [out] edges A pointer to an RDyEdges that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyEdgesCreate(PetscInt num_edges, RDyEdges *edges) {
  PetscFunctionBegin;

  PetscInt cells_per_edge = 2;

  PetscCall(RDyAlloc(PetscInt, num_edges, &edges->ids));
  PetscCall(RDyAlloc(PetscInt, num_edges, &edges->global_ids));
  PetscCall(RDyAlloc(PetscInt, num_edges, &edges->num_cells));
  PetscCall(RDyAlloc(PetscInt, num_edges, &edges->vertex_ids));
  PetscCall(RDyFill(PetscInt, edges->global_ids, num_edges, -1));
  PetscCall(RDyFill(PetscInt, edges->num_cells, num_edges, -1));
  PetscCall(RDyFill(PetscInt, edges->vertex_ids, num_edges, -1));

  PetscCall(RDyAlloc(PetscBool, num_edges, &edges->is_local));
  PetscCall(RDyAlloc(PetscBool, num_edges, &edges->is_internal));

  PetscCall(RDyAlloc(PetscInt, num_edges + 1, &edges->cell_offsets));
  PetscCall(RDyAlloc(PetscInt, num_edges * cells_per_edge, &edges->cell_ids));
  PetscCall(RDyFill(PetscInt, edges->cell_ids, num_edges * cells_per_edge, -1));

  PetscCall(RDyAlloc(RDyPoint, num_edges, &edges->centroids));
  PetscCall(RDyAlloc(RDyVector, num_edges, &edges->normals));
  PetscCall(RDyAlloc(PetscReal, num_edges, &edges->lengths));

  PetscCall(RDyAlloc(PetscReal, num_edges, &edges->cn));
  PetscCall(RDyAlloc(PetscReal, num_edges, &edges->sn));
  PetscCall(RDyFill(PetscReal, edges->cn, num_edges, 0.0));
  PetscCall(RDyFill(PetscReal, edges->sn, num_edges, 0.0));

  for (PetscInt iedge = 0; iedge < num_edges; iedge++) {
    edges->ids[iedge] = iedge;
  }

  for (PetscInt iedge = 0; iedge <= num_edges; iedge++) {
    edges->cell_offsets[iedge] = iedge * cells_per_edge;
  }

  PetscFunctionReturn(0);
}

/// Creates a fully initialized RDyEdges struct from a given DM.
/// @param [in] dm A DM that provides edge data
/// @param [out] edges A pointer to an RDyEdges that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyEdgesCreateFromDM(DM dm, RDyEdges *edges) {
  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscInt eStart, eEnd;
  PetscInt vStart, vEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

  // allocate edge storage
  PetscCall(RDyEdgesCreate(eEnd - eStart, edges));

  PetscInt dim;
  PetscCall(DMGetCoordinateDim(dm, &dim));

  for (PetscInt e = eStart; e < eEnd; e++) {
    PetscInt  iedge = e - eStart;
    PetscReal centroid[dim], normal[dim];
    DMPlexComputeCellGeometryFVM(dm, e, &edges->lengths[iedge], &centroid[0], &normal[0]);

    for (PetscInt idim = 0; idim < dim; idim++) {
      edges->centroids[iedge].X[idim] = centroid[idim];
      edges->normals[iedge].V[idim]   = normal[idim];
    }

    // edge-to-vertex
    PetscInt  pSize;
    PetscInt *p        = NULL;
    PetscInt  use_cone = PETSC_TRUE;
    PetscCall(DMPlexGetTransitiveClosure(dm, e, use_cone, &pSize, &p));
    assert(pSize == 3);
    PetscInt index               = iedge * 2;
    edges->vertex_ids[index + 0] = p[2] - vStart;
    edges->vertex_ids[index + 1] = p[4] - vStart;
    PetscCall(DMPlexRestoreTransitiveClosure(dm, e, use_cone, &pSize, &p));

    // edge-to-cell
    edges->num_cells[iedge] = 0;
    PetscCall(DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &pSize, &p));
    assert(pSize == 2 || pSize == 3);
    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      PetscInt offset        = edges->cell_offsets[iedge];
      PetscInt index         = offset + edges->num_cells[iedge];
      edges->cell_ids[index] = p[i] - cStart;
      edges->num_cells[iedge]++;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &pSize, &p));
  }

  PetscFunctionReturn(0);
}

/// Destroys an RDyEdges struct, freeing its resources.
/// @param [inout] edges An RDyEdges struct whose resources will be freed
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyEdgesDestroy(RDyEdges edges) {
  PetscFunctionBegin;

  PetscCall(RDyFree(edges.ids));
  PetscCall(RDyFree(edges.global_ids));
  PetscCall(RDyFree(edges.is_local));
  PetscCall(RDyFree(edges.num_cells));
  PetscCall(RDyFree(edges.vertex_ids));
  PetscCall(RDyFree(edges.cell_offsets));
  PetscCall(RDyFree(edges.cell_ids));
  PetscCall(RDyFree(edges.is_internal));
  PetscCall(RDyFree(edges.normals));
  PetscCall(RDyFree(edges.centroids));
  PetscCall(RDyFree(edges.lengths));

  PetscFunctionReturn(0);
}

/// a mesh representing a computational domain consisting of a set of cells
/// connected by edges and vertices
typedef struct RDyMesh {
  /// spatial dimension of the mesh (1, 2, or 3)
  PetscInt dim;

  /// number of cells in the mesh (across all processes)
  PetscInt num_cells;
  /// number of cells in the mesh stored on the local process
  PetscInt num_cells_local;
  /// number of edges in the mesh attached to locally stored cells
  PetscInt num_edges;
  /// number of edges that are internal (i.e. shared by two cells)
  PetscInt num_internal_edges;
  /// number of edges that are on the boundary
  PetscInt num_boundary_edges;
  /// number of vertices in the mesh attached to locally stored cells
  PetscInt num_vertices;
  /// number of faces on the domain boundary attached to locally stored cells
  PetscInt num_boundary_faces;

  /// the maximum number of vertices attached to any cell
  PetscInt max_vertex_cells;
  /// the maximum number of vertices attached to any face
  PetscInt max_vertex_faces;

  /// cell information
  RDyCells cells;
  /// vertex information
  RDyVertices vertices;
  /// edge information
  RDyEdges edges;

  /// closure sizes and data for locally stored cells
  PetscInt *closureSize, **closure;
  /// the maximum closure size for any cell (locally stored?)
  PetscInt maxClosureSize;

  /// mapping of global cells to application/natural cells
  PetscInt *nG2A;
} RDyMesh;

/// @brief Computes the cross product of two 3D vectors
/// @param a A RDyVector a
/// @param b A RDyVector b
/// @param c A RDyVector c
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CrossProduct(RDyVector a, RDyVector b, RDyVector *c) {
  PetscFunctionBegin;

  c->V[0] = (a.V[1] * b.V[2] - a.V[2] * b.V[1]);
  c->V[1] = -(a.V[0] * b.V[2] - a.V[2] * b.V[0]);
  c->V[2] = (a.V[0] * b.V[1] - a.V[1] * b.V[0]);

  PetscFunctionReturn(0);
}

/// Computes attributes about an edges needed by RDycore.
/// @param [in] dm A DM that provides edge data
/// @param [inout] mesh A pointer to an RDyMesh mesh data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyComputeAdditionalEdgeAttributes(DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  RDyCells    *cells    = &mesh->cells;
  RDyEdges    *edges    = &mesh->edges;
  RDyVertices *vertices = &mesh->vertices;

  PetscInt cStart, cEnd;
  PetscInt eStart, eEnd;
  PetscInt vStart, vEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

  for (PetscInt e = eStart; e < eEnd; e++) {
    PetscInt iedge = e - eStart;

    PetscInt cellOffset = edges->cell_offsets[iedge];
    PetscInt l          = edges->cell_ids[cellOffset];
    PetscInt r          = edges->cell_ids[cellOffset + 1];

    assert(l >= 0);
    PetscBool is_internal_edge = (r >= 0);

    if (is_internal_edge) {
      mesh->num_internal_edges++;
    } else {
      mesh->num_boundary_edges++;
    }

    /*
                 Case-1                      Case-2                       Update Case-2

                    v2                         v2                             v1
                   /|\                        /|\                             |
                    |                          |                              |
                    |---> normal               | ----> normal     normal <----|
                    |                          |                              |
             L -----|-----> R          R <-----|----- L               R <-----|----- L
                    |                          |                              |
                    |                          |                              |
                    |                          |                             \|/
                    v1                         v1                             v2

    In DMPlex, the cross product of the normal vector to the edge and vector joining the
    vertices of the edge (i.e. v1Tov2)  always points in the positive z-direction.
    However, the vector joining the left and the right cell may not be in the same direction
    as the normal vector to the edge (Case-2). Thus, the edge information in the Case-2 is
    updated by spawing the vertex ids and flipping the edge normal.
    */

    PetscInt v_offset = iedge * 2;
    PetscInt vid_1    = edges->vertex_ids[v_offset + 0];
    PetscInt vid_2    = edges->vertex_ids[v_offset + 1];

    RDyVector edge_parallel;  // a vector parallel along the edge in 2D
    for (PetscInt idim = 0; idim < 2; idim++) {
      edge_parallel.V[idim] = vertices->points[vid_2].X[idim] - vertices->points[vid_1].X[idim];
    }
    edge_parallel.V[2] = 0.0;

    // In case of an internal edge, a vector from the left cell to the right cell.
    // In case of a boundary edge, a vector from the left cell to edge centroid.
    // Note: This is a vector in 2D.
    RDyVector vec_L2RorEC;

    if (is_internal_edge) {
      for (PetscInt idim = 0; idim < 2; idim++) {
        vec_L2RorEC.V[idim] = cells->centroids[r].X[idim] - cells->centroids[l].X[idim];
      }

    } else {
      for (PetscInt idim = 0; idim < 2; idim++) {
        vec_L2RorEC.V[idim] = (vertices->points[vid_2].X[idim] + vertices->points[vid_1].X[idim]) / 2.0 - cells->centroids[l].X[idim];
      }
    }
    vec_L2RorEC.V[2] = 0.0;

    // Compute a vector perpendicular to the edge_parallel vector via a clockwise
    // 90 degree rotation
    RDyVector edge_perp;
    edge_perp.V[0] = edge_parallel.V[1];
    edge_perp.V[1] = -edge_parallel.V[0];

    // Compute the dot product to check if vector joining L-to-R is pointing
    // in the direction of the vector perpendicular to the edge.
    PetscReal dot_prod = vec_L2RorEC.V[0] * edge_perp.V[0] + vec_L2RorEC.V[1] * edge_perp.V[1];

    if (dot_prod < 0.0) {
      // The angle between edge_perp and vec_L2RorEC is greater than 90 deg.
      // Thus, flip vertex ids and the normal vector
      edges->vertex_ids[v_offset + 0] = vid_2;
      edges->vertex_ids[v_offset + 1] = vid_1;
      for (PetscInt idim = 0; idim < 3; idim++) {
        edges->normals[iedge].V[idim] *= -1.0;
      }
    }

    vid_1 = edges->vertex_ids[v_offset + 0];
    vid_2 = edges->vertex_ids[v_offset + 1];

    PetscReal x1 = vertices->points[vid_1].X[0];
    PetscReal y1 = vertices->points[vid_1].X[1];
    PetscReal x2 = vertices->points[vid_2].X[0];
    PetscReal y2 = vertices->points[vid_2].X[1];

    PetscReal dx = x2 - x1;
    PetscReal dy = y2 - y1;
    PetscReal ds = PetscSqrtReal(Square(dx) + Square(dy));

    edges->sn[iedge] = -dx / ds;
    edges->cn[iedge] = dy / ds;
  }

  // allocate memory to save IDs of internal and boundary edges
  PetscCall(RDyAlloc(PetscInt, mesh->num_internal_edges, &edges->internal_edge_ids));
  PetscCall(RDyAlloc(PetscInt, mesh->num_boundary_edges, &edges->boundary_edge_ids));

  // now save the IDs
  mesh->num_internal_edges = 0;
  mesh->num_boundary_edges = 0;

  for (PetscInt e = eStart; e < eEnd; e++) {
    PetscInt iedge      = e - eStart;
    PetscInt cellOffset = edges->cell_offsets[iedge];
    PetscInt l          = edges->cell_ids[cellOffset];
    PetscInt r          = edges->cell_ids[cellOffset + 1];

    if (r >= 0 && l >= 0) {
      edges->internal_edge_ids[mesh->num_internal_edges++] = iedge;
    } else {
      edges->boundary_edge_ids[mesh->num_boundary_edges++] = iedge;
    }
  }

  PetscFunctionReturn(0);
}

/// @brief Checks if the vertices forming the triangle are in counter clockwise direction
/// @param [in] xyz0 Coordinates of the first vertex of the triangle
/// @param [in] xyz1 Coordinates of the second vertex of the triangle
/// @param [in] xyz2 Coordinates of the third vertex of the triangle
/// @return 1 if the vertices are in counter clockwise direction, otherwise 0
PetscBool AreVerticesOrientedCounterClockwise(PetscReal xyz0[3], PetscReal xyz1[3], PetscReal xyz2[3]) {
  PetscFunctionBegin;

  PetscBool result = PETSC_TRUE;

  PetscReal x0, y0;
  PetscReal x1, y1;
  PetscReal x2, y2;

  x0 = xyz0[0];
  y0 = xyz0[1];
  x1 = xyz1[0];
  y1 = xyz1[1];
  x2 = xyz2[0];
  y2 = xyz2[1];

  PetscFunctionReturn((y1 - y0) * (x2 - x1) - (y2 - y1) * (x1 - x0) < 0);

  PetscFunctionReturn(result);
}

/// @brief Computes slope in x and y direction for a triangle (i.e. dz/dx and dz/dy)
/// @param [in] xyz0 Coordinates of the first vertex of the triangle
/// @param [in] xyz1 Coordinates of the second vertex of the triangle
/// @param [in] xyz2 Coordinates of the third vertex of the triangle
/// @param [out] *dz_dx Slope in x-direction
/// @param [out] dz_dy Slope in y-direction
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ComputeXYSlopesForTriangle(PetscReal xyz0[3], PetscReal xyz1[3], PetscReal xyz2[3], PetscReal *dz_dx, PetscReal *dz_dy) {
  PetscFunctionBegin;

  PetscReal x0, y0, z0;
  PetscReal x1, y1, z1;
  PetscReal x2, y2, z2;

  x0 = xyz0[0];
  y0 = xyz0[1];
  z0 = xyz0[2];

  if (AreVerticesOrientedCounterClockwise(xyz0, xyz1, xyz2)) {
    x1 = xyz1[0];
    y1 = xyz1[1];
    z1 = xyz1[2];
    x2 = xyz2[0];
    y2 = xyz2[1];
    z2 = xyz2[2];
  } else {
    x1 = xyz2[0];
    y1 = xyz2[1];
    z1 = xyz2[2];
    x2 = xyz1[0];
    y2 = xyz1[1];
    z2 = xyz1[2];
  }

  PetscReal num, den;
  num    = (y2 - y0) * (z1 - z0) - (y1 - y0) * (z2 - z0);
  den    = (y2 - y0) * (x1 - x0) - (y1 - y0) * (x2 - x0);
  *dz_dx = num / den;

  num    = (x2 - x0) * (z1 - z0) - (x1 - x0) * (z2 - z0);
  den    = (x2 - x0) * (y1 - y0) - (x1 - x0) * (y2 - y0);
  *dz_dy = num / den;

  PetscFunctionReturn(0);
}

/// Computes geometric attributes about a cell needed by RDycore.
/// @param [inout] mesh A pointer to an RDyMesh mesh data.
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyComputeAdditionalCellAttributes(RDyMesh *mesh) {
  PetscFunctionBegin;

  RDyCells    *cells    = &mesh->cells;
  RDyVertices *vertices = &mesh->vertices;

  for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
    PetscInt nverts = cells->num_vertices[icell];

    if (nverts == 3) {
      PetscInt offset = cells->vertex_offsets[icell];
      PetscInt v0     = cells->vertex_ids[offset + 0];
      PetscInt v1     = cells->vertex_ids[offset + 1];
      PetscInt v2     = cells->vertex_ids[offset + 2];

      PetscCall(ComputeXYSlopesForTriangle(vertices->points[v0].X, vertices->points[v1].X, vertices->points[v2].X, &cells->dz_dx[icell],
                                           &cells->dz_dy[icell]));

    } else if (nverts == 4) {
      PetscInt offset = cells->vertex_offsets[icell];
      PetscInt v0     = cells->vertex_ids[offset + 0];
      PetscInt v1     = cells->vertex_ids[offset + 1];
      PetscInt v2     = cells->vertex_ids[offset + 2];
      PetscInt v3     = cells->vertex_ids[offset + 3];

      PetscInt vertexIDs[4][2];
      vertexIDs[0][0] = v0;
      vertexIDs[0][1] = v1;
      vertexIDs[1][0] = v1;
      vertexIDs[1][1] = v2;
      vertexIDs[2][0] = v2;
      vertexIDs[2][1] = v3;
      vertexIDs[3][0] = v3;
      vertexIDs[3][1] = v0;

      PetscReal dz_dx, dz_dy;
      cells->dz_dx[icell] = 0.0;
      cells->dz_dy[icell] = 0.0;

      // TODO: Revisit the approach to compute dz/dx and dz/y for quad cells.
      for (PetscInt ii = 0; ii < 4; ii++) {
        PetscInt a = vertexIDs[ii][0];
        PetscInt b = vertexIDs[ii][1];

        PetscCall(ComputeXYSlopesForTriangle(vertices->points[a].X, vertices->points[b].X, cells->centroids[icell].X, &dz_dx, &dz_dy));
        cells->dz_dx[icell] += 0.5 * dz_dx;
        cells->dz_dy[icell] += 0.5 * dz_dy;
      }

    } else {
      printf("The code only support cells with 3 or 4 vertices, but found a cell with num of vertices = %d\n", nverts);
      exit(0);
    }
  }

  PetscFunctionReturn(0);
}

static PetscErrorCode SaveNaturalCellIDs(DM dm, RDyCells *cells, PetscInt rank) {
  PetscFunctionBegin;

  PetscBool useNatural;
  PetscCall(DMGetUseNatural(dm, &useNatural));

  if (useNatural) {
    PetscInt num_fields;

    PetscCall(DMGetNumFields(dm, &num_fields));

    // Create the natural vector
    Vec      natural;
    PetscInt natural_size = 0, natural_start;
    PetscCall(DMPlexCreateNaturalVector(dm, &natural));
    PetscCall(PetscObjectSetName((PetscObject)natural, "Natural Vec"));
    PetscCall(VecGetLocalSize(natural, &natural_size));
    PetscCall(VecGetOwnershipRange(natural, &natural_start, NULL));

    // Add entries in the natural vector
    PetscScalar *entries;
    PetscCall(VecGetArray(natural, &entries));
    for (PetscInt i = 0; i < natural_size; ++i) {
      if (i % num_fields == 0) {
        entries[i] = (natural_start + i) / num_fields;
      } else {
        entries[i] = -(rank + 1);
      }
    }
    PetscCall(VecRestoreArray(natural, &entries));

    // Map natural IDs in global order
    Vec global;
    PetscCall(DMCreateGlobalVector(dm, &global));
    PetscCall(PetscObjectSetName((PetscObject)global, "Global Vec"));
    PetscCall(DMPlexNaturalToGlobalBegin(dm, natural, global));
    PetscCall(DMPlexNaturalToGlobalEnd(dm, natural, global));

    // Map natural IDs in local order
    Vec         local;
    PetscViewer selfviewer;
    PetscCall(DMCreateLocalVector(dm, &local));
    PetscCall(PetscObjectSetName((PetscObject)local, "Local Vec"));
    PetscCall(DMGlobalToLocalBegin(dm, global, INSERT_VALUES, local));
    PetscCall(DMGlobalToLocalEnd(dm, global, INSERT_VALUES, local));
    PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &selfviewer));
    PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &selfviewer));

    // Save natural IDs
    PetscInt local_size;
    PetscCall(VecGetLocalSize(local, &local_size));
    PetscCall(VecGetArray(local, &entries));
    for (PetscInt i = 0; i < local_size / num_fields; ++i) {
      cells->natural_ids[i] = entries[i * num_fields];
    }
    PetscCall(VecRestoreArray(local, &entries));

    // Cleanup
    PetscCall(VecDestroy(&natural));
    PetscCall(VecDestroy(&global));
    PetscCall(VecDestroy(&local));
  }

  PetscFunctionReturn(0);
}

/// Creates an RDyMesh from a PETSc DM.
/// @param [in] dm A PETSc DM
/// @param [out] mesh A pointer to an RDyMesh that stores allocated data.
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyMeshCreateFromDM(DM dm, RDyMesh *mesh) {
  PetscFunctionBegin;

  // Determine the number of cells in the mesh
  PetscInt cStart, cEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  mesh->num_cells = cEnd - cStart;

  // Determine the number of edges in the mesh
  PetscInt eStart, eEnd;
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  mesh->num_edges          = eEnd - eStart;
  mesh->num_internal_edges = 0;
  mesh->num_boundary_edges = 0;

  // Determine the number of vertices in the mesh
  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);
  mesh->num_vertices = vEnd - vStart;

  // Create mesh elements from the DM
  PetscCall(RDyCellsCreateFromDM(dm, &mesh->cells));
  PetscCall(RDyEdgesCreateFromDM(dm, &mesh->edges));
  PetscCall(RDyVerticesCreateFromDM(dm, &mesh->vertices));
  PetscCall(RDyComputeAdditionalEdgeAttributes(dm, mesh));
  PetscCall(RDyComputeAdditionalCellAttributes(mesh));

  // Count up local cells.
  mesh->num_cells_local = 0;
  for (PetscInt icell = 0; icell < mesh->num_cells; ++icell) {
    if (mesh->cells.is_local[icell]) {
      ++mesh->num_cells_local;
    }
  }

  // Extract natural cell IDs from the DM.
  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscInt rank;
  MPI_Comm_rank(comm, &rank);
  PetscCall(SaveNaturalCellIDs(dm, &mesh->cells, rank));

  PetscFunctionReturn(0);
}

/// Destroys an RDyMesh struct, freeing its resources.
/// @param [inout] edges An RDyMesh struct whose resources will be freed
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyMeshDestroy(RDyMesh mesh) {
  PetscFunctionBegin;
  PetscCall(RDyCellsDestroy(mesh.cells));
  PetscCall(RDyEdgesDestroy(mesh.edges));
  PetscCall(RDyVerticesDestroy(mesh.vertices));
  PetscCall(RDyFree(mesh.nG2A));
  PetscFunctionReturn(0);
}

/// an application context that stores data relevant to a simulation
struct _n_RDyApp {
  /// MPI communicator used for the simulation
  MPI_Comm comm;
  /// MPI rank of local process
  PetscInt rank;
  /// Number of processes in the communicator
  PetscInt comm_size;
  /// filename storing input data for the simulation
  char filename[PETSC_MAX_PATH_LEN];
  /// filename storing initial condition for the simulation
  char initial_condition_filename[PETSC_MAX_PATH_LEN];
  /// PETSc grid
  DM dm;
  /// A DM for creating PETSc Vecs with 1 DOF
  DM auxdm;
  /// Number of cells in the x direction
  PetscInt Nx;
  /// Number of cells in the y direction
  PetscInt Ny;
  /// grid spacing in the x direction
  PetscReal dx;
  /// grid spacing in the y direction
  PetscReal dy;
  /// domain extent in x
  PetscReal Lx;
  /// domain extent in y
  PetscReal Ly;
  /// water depth for the upstream of dam [m]
  PetscReal hu;
  /// water depth for the downstream of dam [m]
  PetscReal hd;
  /// water depth below which no horizontal flow occurs
  PetscReal tiny_h;
  /// total number of time steps
  PetscInt Nt;
  /// time step size
  PetscReal dt;
  /// index of current timestep
  PetscInt tstep;

  PetscInt  ndof;
  Vec       B, localB;
  Vec       localX;
  PetscBool debug, save, add_building;
  PetscBool interpolate;

  /// mesh representing simulation domain
  RDyMesh mesh;
};

/// alias for pointer to the application context
typedef struct _n_RDyApp *RDyApp;

PetscErrorCode RDyAllocate_IntegerArray_1D(PetscInt **array_1D, PetscInt ndim_1) {
  PetscFunctionBegin;
  PetscCall(RDyAlloc(PetscInt, ndim_1, array_1D));
  PetscCall(RDyFill(PetscInt, *array_1D, ndim_1, -1));
  PetscFunctionReturn(0);
}

/// @brief Process command line options
/// @param [in] comm A MPI commmunicator
/// @param [inout] app An application context to be updated
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ProcessOptions(MPI_Comm comm, RDyApp app) {
  PetscFunctionBegin;

  app->comm   = comm;
  app->Nx     = 4;
  app->Ny     = 5;
  app->dx     = 1.0;
  app->dy     = 1.0;
  app->hu     = 10.0;  // water depth for the upstream of dam   [m]
  app->hd     = 5.0;   // water depth for the downstream of dam [m]
  app->tiny_h = 1e-7;
  app->ndof   = 3;

  MPI_Comm_size(app->comm, &app->comm_size);
  MPI_Comm_rank(app->comm, &app->rank);

  PetscOptionsBegin(app->comm, NULL, "2D Mesh Options", "");
  {
    PetscCall(PetscOptionsInt("-Nx", "Number of cells in X", "", app->Nx, &app->Nx, NULL));
    PetscCall(PetscOptionsInt("-Ny", "Number of cells in Y", "", app->Ny, &app->Ny, NULL));
    PetscCall(PetscOptionsInt("-Nt", "Number of time steps", "", app->Nt, &app->Nt, NULL));
    PetscCall(PetscOptionsReal("-dx", "dx", "", app->dx, &app->dx, NULL));
    PetscCall(PetscOptionsReal("-dy", "dy", "", app->dy, &app->dy, NULL));
    PetscCall(PetscOptionsReal("-hu", "hu", "", app->hu, &app->hu, NULL));
    PetscCall(PetscOptionsReal("-hd", "hd", "", app->hd, &app->hd, NULL));
    PetscCall(PetscOptionsReal("-dt", "dt", "", app->dt, &app->dt, NULL));
    PetscCall(PetscOptionsBool("-b", "Add buildings", "", app->add_building, &app->add_building, NULL));
    PetscCall(PetscOptionsBool("-debug", "debug", "", app->debug, &app->debug, NULL));
    PetscCall(PetscOptionsBool("-save", "save outputs", "", app->save, &app->save, NULL));
    PetscCall(PetscOptionsString("-mesh_filename", "The mesh file", "ex2.c", app->filename, app->filename, PETSC_MAX_PATH_LEN, NULL));
    PetscCall(PetscOptionsString("-initial_condition_filename", "The initial condition file", "ex2.c", app->initial_condition_filename, app->initial_condition_filename, PETSC_MAX_PATH_LEN, NULL));
  }
  PetscOptionsEnd();

  assert(app->hu >= 0.);
  assert(app->hd >= 0.);

  app->Lx = app->Nx * app->dx;
  app->Ly = app->Ny * app->dy;

  PetscReal max_time = app->Nt * app->dt;
  PetscPrintf(app->comm, "Max simulation time is %f\n", max_time);

  PetscFunctionReturn(0);
}

/// Creates the PETSc DM as a box or from a file. Add three DOFs and distribute the DM
/// @param [inout] app A app data structure that is modified
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateDM(RDyApp app) {
  PetscFunctionBegin;

  size_t len;

  PetscStrlen(app->filename, &len);
  if (!len) {
    PetscInt  dim     = 2;
    PetscInt  faces[] = {app->Nx, app->Ny};
    PetscReal lower[] = {0.0, 0.0};
    PetscReal upper[] = {app->Lx, app->Ly};

    PetscCall(DMPlexCreateBoxMesh(app->comm, dim, PETSC_FALSE, faces, lower, upper, PETSC_NULL, PETSC_TRUE, &app->dm));
  } else {
    DMPlexCreateFromFile(app->comm, app->filename, "ex2.c", PETSC_FALSE, &app->dm);
  }

  DM dmInterp;
  PetscCall(DMPlexInterpolate(app->dm, &dmInterp));
  PetscCall(DMDestroy(&app->dm));
  app->dm = dmInterp;

  PetscCall(DMPlexDistributeSetDefault(app->dm, PETSC_FALSE));

  PetscObjectSetName((PetscObject)app->dm, "Mesh");
  PetscCall(DMSetFromOptions(app->dm));
  PetscCall(DMViewFromOptions(app->dm, NULL, "-orig_dm_view"));

  // Determine the number of cells, edges, and vertices of the mesh
  PetscInt cStart, cEnd;
  DMPlexGetHeightStratum(app->dm, 0, &cStart, &cEnd);

  // Create a single section that has 3 DOFs
  PetscSection sec;
  PetscCall(PetscSectionCreate(app->comm, &sec));

  // Add the 3 DOFs
  PetscInt nfield             = 3;
  PetscInt num_field_dof[]    = {1, 1, 1};
  char     field_names[3][20] = {{"Height"}, {"Momentum in x-dir"}, {"Momentum in y-dir"}};

  nfield = 3;
  PetscCall(PetscSectionSetNumFields(sec, nfield));
  PetscInt total_num_dof = 0;
  for (PetscInt ifield = 0; ifield < nfield; ifield++) {
    PetscCall(PetscSectionSetFieldName(sec, ifield, &field_names[ifield][0]));
    PetscCall(PetscSectionSetFieldComponents(sec, ifield, num_field_dof[ifield]));
    total_num_dof += num_field_dof[ifield];
  }

  PetscCall(PetscSectionSetChart(sec, cStart, cEnd));
  for (PetscInt c = cStart; c < cEnd; c++) {
    for (PetscInt ifield = 0; ifield < nfield; ifield++) {
      PetscCall(PetscSectionSetFieldDof(sec, c, ifield, num_field_dof[ifield]));
    }
    PetscCall(PetscSectionSetDof(sec, c, total_num_dof));
  }
  PetscCall(PetscSectionSetUp(sec));
  PetscCall(DMSetLocalSection(app->dm, sec));
  PetscCall(PetscSectionViewFromOptions(sec, NULL, "-layout_view"));
  PetscCall(PetscSectionDestroy(&sec));
  PetscCall(DMSetBasicAdjacency(app->dm, PETSC_TRUE, PETSC_TRUE));

  // Before distributing the DM, set a flag to create mapping from natural-to-local order
  PetscCall(DMSetUseNatural(app->dm, PETSC_TRUE));

  // Distrubte the DM
  DM dmDist;
  PetscCall(DMPlexDistribute(app->dm, 1, NULL, &dmDist));
  if (dmDist) {
    DMDestroy(&app->dm);
    app->dm = dmDist;
  }

  PetscCall(DMViewFromOptions(app->dm, NULL, "-dm_view"));

  PetscFunctionReturn(0);
}

/// Creates an auxillary PETSc DM for 1 DOFs from the main DM
/// @param [inout] app A app data structure that is modified
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateAuxDM(RDyApp app) {
  PetscFunctionBegin;

  PetscCall(DMClone(app->dm, &app->auxdm));

  PetscSection auxsec;
  PetscCall(PetscSectionCreate(app->comm, &auxsec));

  PetscInt aux_nfield             = 1;
  PetscInt aux_num_field_dof[]    = {1};
  char     aux_field_names[1][20] = {{"Parameter"}};
  PetscCall(PetscSectionSetNumFields(auxsec, aux_nfield));
  PetscInt aux_total_num_dof = 0;
  for (PetscInt ifield = 0; ifield < aux_nfield; ifield++) {
    PetscCall(PetscSectionSetFieldName(auxsec, ifield, &aux_field_names[ifield][0]));
    PetscCall(PetscSectionSetFieldComponents(auxsec, ifield, aux_num_field_dof[ifield]));
    aux_total_num_dof += aux_num_field_dof[ifield];
  }

  // Determine the number of cells, edges, and vertices of the mesh
  PetscInt cStart, cEnd;
  DMPlexGetHeightStratum(app->dm, 0, &cStart, &cEnd);

  PetscCall(PetscSectionSetChart(auxsec, cStart, cEnd));
  for (PetscInt c = cStart; c < cEnd; c++) {
    for (PetscInt ifield = 0; ifield < aux_nfield; ifield++) {
      PetscCall(PetscSectionSetFieldDof(auxsec, c, ifield, aux_num_field_dof[ifield]));
    }
    PetscCall(PetscSectionSetDof(auxsec, c, aux_total_num_dof));
  }
  PetscCall(DMSetLocalSection(app->auxdm, auxsec));
  PetscCall(PetscSectionViewFromOptions(auxsec, NULL, "-aux_layout_view"));
  PetscCall(PetscSectionSetUp(auxsec));

  PetscSF sfMigration, sfNatural;
  DMPlexGetMigrationSF(app->dm, &sfMigration);
  DMPlexCreateGlobalToNaturalSF(app->auxdm, auxsec, sfMigration, &sfNatural);
  DMPlexSetGlobalToNaturalSF(app->auxdm, sfNatural);
  PetscSFDestroy(&sfNatural);

  PetscFunctionReturn(0);
}

/// @brief Sets initial condition for [h, hu, hv]
/// @param [in] app An application context
/// @param [inout] X Vec for initial condition
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode SetInitialCondition(RDyApp app, Vec X) {
  PetscFunctionBegin;

  RDyMesh  *mesh  = &app->mesh;
  RDyCells *cells = &mesh->cells;

  PetscCall(VecZeroEntries(X));

  PetscScalar *x_ptr;
  VecGetArray(X, &x_ptr);

  for (PetscInt icell = 0; icell < mesh->num_cells_local; icell++) {
    PetscInt ndof = app->ndof;
    PetscInt idx  = icell * ndof;
    if (cells->centroids[icell].X[1] < 95.0) {
      x_ptr[idx] = app->hu;
    } else {
      x_ptr[idx] = app->hd;
    }
  }

  VecRestoreArray(X, &x_ptr);

  PetscFunctionReturn(0);
}

/// @brief Reads initial condition for [h, hu, hv] from file
/// @param [in] app An application context
/// @param [inout] X Vec for initial condition
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode SetInitialConditionFromFile(RDyApp app, Vec X) {
  PetscFunctionBegin;

  PetscCall(VecZeroEntries(X));

  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(app->comm, app->initial_condition_filename, FILE_MODE_READ, &viewer));
  Vec natural;
  PetscCall(DMPlexCreateNaturalVector(app->dm, &natural));
  PetscCall(VecLoad(natural, viewer));
  PetscCall(DMPlexNaturalToGlobalBegin(app->dm, natural, X));
  PetscCall(DMPlexNaturalToGlobalEnd(app->dm, natural, X));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&natural));

  PetscFunctionReturn(0);
}

PetscErrorCode AddBuildings(RDyApp app) {
  PetscFunctionBeginUser;

  PetscInt bu = 30 / app->dx;
  PetscInt bd = 105 / app->dx;
  PetscInt bl = 95 / app->dy;
  PetscInt br = 105 / app->dy;

  PetscCall(VecZeroEntries(app->B));

  PetscScalar *b_ptr;
  PetscCall(VecGetArray(app->B, &b_ptr));

  /*

  x represents the reflecting wall,
  o represents the dam that will be broken suddenly.
  hu is the upsatream water dpeth, and hd is the downstream water depth.


/|\             xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
 |              x                   xxxx                 x
 |              x                   xxxx                 x
95[m]           x                   xxxx                 x
 |              x                   xxxx                 x
 |              x                   xxxx                 x
 |              x                   xxxx                 x
\|/             x                   xxxx                 x
/|\             x        hu         o          hd        x
 |              x                   o                    x
 |              x                   o                    x
105[m]          x                   o                    x
 |    /|\       x                   xxxx                 x
 |     |        x                   xxxx                 x
 |    30[m]     x                   xxxx                 x
 |     |        x                   xxxx                 x
 |     |        x                   xxxx                 x
\|/   \|/       xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
                |                   |  |                 |
                |<-----  95[m] ---->|  |                 |
                |<------- 105[m] ----->|                 |
                |<---------------- 200[m] -------------->|
  */

  RDyMesh  *mesh  = &app->mesh;
  RDyCells *cells = &mesh->cells;

  PetscInt nbnd_1 = 0, nbnd_2 = 0;
  for (PetscInt icell = 0; icell < mesh->num_cells_local; icell++) {
    PetscReal xc = cells->centroids[icell].X[1];
    PetscReal yc = cells->centroids[icell].X[0];
    if (yc < bu && xc >= bl && xc < br) {
      b_ptr[icell] = 1.0;
      nbnd_1++;
    } else if (yc >= bd && xc >= bl && xc < br) {
      b_ptr[icell] = 1.0;
      nbnd_2++;
    }
  }

  PetscCall(VecRestoreArray(app->B, &b_ptr));

  PetscCall(DMGlobalToLocalBegin(app->auxdm, app->B, INSERT_VALUES, app->localB));
  PetscCall(DMGlobalToLocalEnd(app->auxdm, app->B, INSERT_VALUES, app->localB));

  PetscCall(PetscPrintf(app->comm, "Building size: bu=%d,bd=%d,bl=%d,br=%d\n", bu, bd, bl, br));
  PetscCall(PetscPrintf(app->comm, "Buildings added sucessfully!\n"));

  PetscViewer viewer;
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF, "outputs/localB.dat", FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(app->localB, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "outputs/B.dat", FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(app->B, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscFunctionReturn(0);
}

/// @brief Computes flux based on Roe solver
/// @param N Size of the array
/// @param [in] hl Height left of the edge
/// @param [in] hr Height right of the edge
/// @param [in] ul Velocity in x-dir left of the edge
/// @param [in] ur Velocity in x-dir right of the edge
/// @param [in] vl Velocity in y-dir left of the edge
/// @param [in] vr Velocity in y-dir right of the edge
/// @param [in] sn sine of the angle between edge and y-axis
/// @param [in] cn cosine of the angle between edge and y-axis
/// @param [out] fij flux
/// @param [out] amax maximum wave speed
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode solver(PetscInt N, const PetscReal hl[N], const PetscReal hr[N], const PetscReal ul[N], const PetscReal ur[N], const PetscReal vl[N],
                      const PetscReal vr[N], const PetscReal sn[N], const PetscReal cn[N], PetscReal fij[N][3], PetscReal amax[N]) {
  PetscFunctionBeginUser;

  PetscReal grav = 9.806;

  for (PetscInt n = 0; n < N; n++) {
    // Compute Roe averages
    PetscReal duml  = pow(hl[n], 0.5);
    PetscReal dumr  = pow(hr[n], 0.5);
    PetscReal cl    = pow(grav * hl[n], 0.5);
    PetscReal cr    = pow(grav * hr[n], 0.5);
    PetscReal hhat  = duml * dumr;
    PetscReal uhat  = (duml * ul[n] + dumr * ur[n]) / (duml + dumr);
    PetscReal vhat  = (duml * vl[n] + dumr * vr[n]) / (duml + dumr);
    PetscReal chat  = pow(0.5 * grav * (hl[n] + hr[n]), 0.5);
    PetscReal uperp = uhat * cn[n] + vhat * sn[n];

    PetscReal dh     = hr[n] - hl[n];
    PetscReal du     = ur[n] - ul[n];
    PetscReal dv     = vr[n] - vl[n];
    PetscReal dupar  = -du * sn[n] + dv * cn[n];
    PetscReal duperp = du * cn[n] + dv * sn[n];

    PetscReal dW[3];
    dW[0] = 0.5 * (dh - hhat * duperp / chat);
    dW[1] = hhat * dupar;
    dW[2] = 0.5 * (dh + hhat * duperp / chat);

    PetscReal uperpl = ul[n] * cn[n] + vl[n] * sn[n];
    PetscReal uperpr = ur[n] * cn[n] + vr[n] * sn[n];
    PetscReal al1    = uperpl - cl;
    PetscReal al3    = uperpl + cl;
    PetscReal ar1    = uperpr - cr;
    PetscReal ar3    = uperpr + cr;

    PetscReal R[3][3];
    R[0][0] = 1.0;
    R[0][1] = 0.0;
    R[0][2] = 1.0;
    R[1][0] = uhat - chat * cn[n];
    R[1][1] = -sn[n];
    R[1][2] = uhat + chat * cn[n];
    R[2][0] = vhat - chat * sn[n];
    R[2][1] = cn[n];
    R[2][2] = vhat + chat * sn[n];

    PetscReal da1 = fmax(0.0, 2.0 * (ar1 - al1));
    PetscReal da3 = fmax(0.0, 2.0 * (ar3 - al3));
    PetscReal a1  = fabs(uperp - chat);
    PetscReal a2  = fabs(uperp);
    PetscReal a3  = fabs(uperp + chat);

    // Critical flow fix
    if (a1 < da1) {
      a1 = 0.5 * (a1 * a1 / da1 + da1);
    }
    if (a3 < da3) {
      a3 = 0.5 * (a3 * a3 / da3 + da3);
    }

    // Compute interface flux
    PetscReal A[3][3];
    for (PetscInt i = 0; i < 3; i++) {
      for (PetscInt j = 0; j < 3; j++) {
        A[i][j] = 0.0;
      }
    }
    A[0][0] = a1;
    A[1][1] = a2;
    A[2][2] = a3;

    PetscReal FL[3], FR[3];
    FL[0] = uperpl * hl[n];
    FL[1] = ul[n] * uperpl * hl[n] + 0.5 * grav * hl[n] * hl[n] * cn[n];
    FL[2] = vl[n] * uperpl * hl[n] + 0.5 * grav * hl[n] * hl[n] * sn[n];

    FR[0] = uperpr * hr[n];
    FR[1] = ur[n] * uperpr * hr[n] + 0.5 * grav * hr[n] * hr[n] * cn[n];
    FR[2] = vr[n] * uperpr * hr[n] + 0.5 * grav * hr[n] * hr[n] * sn[n];

    // fij = 0.5*(FL + FR - matmul(R,matmul(A,dW))
    fij[n][0] = 0.5 * (FL[0] + FR[0] - R[0][0] * A[0][0] * dW[0] - R[0][1] * A[1][1] * dW[1] - R[0][2] * A[2][2] * dW[2]);
    fij[n][1] = 0.5 * (FL[1] + FR[1] - R[1][0] * A[0][0] * dW[0] - R[1][1] * A[1][1] * dW[1] - R[1][2] * A[2][2] * dW[2]);
    fij[n][2] = 0.5 * (FL[2] + FR[2] - R[2][0] * A[0][0] * dW[0] - R[2][1] * A[1][1] * dW[1] - R[2][2] * A[2][2] * dW[2]);

    amax[n] = chat + fabs(uperp);
  }

  PetscFunctionReturn(0);
}

/// @brief Computes velocities in x and y-dir based on momentum in x and y-dir
/// @param N Size of the array
/// @param tiny_h Threshold value for height
/// @param h Height
/// @param hu Momentum in x-dir
/// @param hv Momentum in y-dir
/// @param u Velocity in x-dir
/// @param v Velocity in y-dir
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode GetVelocityFromMomentum(PetscInt N, PetscReal tiny_h, const PetscReal h[N], const PetscReal hu[N], const PetscReal hv[N],
                                              PetscReal u[N], PetscReal v[N]) {
  PetscFunctionBeginUser;

  for (PetscInt n = 0; n < N; n++) {
    if (h[n] < tiny_h) {
      u[n] = 0.0;
      v[n] = 0.0;
    } else {
      u[n] = hu[n] / h[n];
      v[n] = hv[n] / h[n];
    }
  }

  PetscFunctionReturn(0);
}

/// @brief It computes RHSFunction for internal edges
/// @param [inout] app A RDyApp struct
/// @param [inout] F A global flux Vec
PetscErrorCode RHSFunctionForInternalEdges(RDyApp app, Vec F, PetscReal *amax_value) {
  PetscFunctionBeginUser;

  RDyMesh  *mesh  = &app->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  // Get pointers to vector data
  PetscScalar *x_ptr, *f_ptr, *b_ptr;
  PetscCall(VecGetArray(app->localX, &x_ptr));
  PetscCall(VecGetArray(F, &f_ptr));
  PetscCall(VecGetArray(app->localB, &b_ptr));

  PetscInt  ndof = app->ndof;
  PetscInt  num  = mesh->num_internal_edges;
  PetscReal hl_vec_int[num], hul_vec_int[num], hvl_vec_int[num], ul_vec_int[num], vl_vec_int[num];
  PetscReal hr_vec_int[num], hur_vec_int[num], hvr_vec_int[num], ur_vec_int[num], vr_vec_int[num];
  PetscReal sn_vec_int[num], cn_vec_int[num];
  PetscReal flux_vec_int[num][3], amax_vec_int[num];

  // Collect the h/hu/hv for left and right cells to compute u/v
  for (PetscInt ii = 0; ii < mesh->num_internal_edges; ii++) {
    PetscInt iedge      = edges->internal_edge_ids[ii];
    PetscInt cellOffset = edges->cell_offsets[iedge];
    PetscInt l          = edges->cell_ids[cellOffset];
    PetscInt r          = edges->cell_ids[cellOffset + 1];

    hl_vec_int[ii]  = x_ptr[l * ndof + 0];
    hul_vec_int[ii] = x_ptr[l * ndof + 1];
    hvl_vec_int[ii] = x_ptr[l * ndof + 2];

    hr_vec_int[ii]  = x_ptr[r * ndof + 0];
    hur_vec_int[ii] = x_ptr[r * ndof + 1];
    hvr_vec_int[ii] = x_ptr[r * ndof + 2];
  }

  // Compute u/v for left and right cells
  PetscCall(GetVelocityFromMomentum(num, app->tiny_h, hl_vec_int, hul_vec_int, hvl_vec_int, ul_vec_int, vl_vec_int));
  PetscCall(GetVelocityFromMomentum(num, app->tiny_h, hr_vec_int, hur_vec_int, hvr_vec_int, ur_vec_int, vr_vec_int));

  // Update u/v for reflective internal edges
  for (PetscInt ii = 0; ii < mesh->num_internal_edges; ii++) {
    PetscInt  iedge      = edges->internal_edge_ids[ii];
    PetscInt  cellOffset = edges->cell_offsets[iedge];
    PetscInt  l          = edges->cell_ids[cellOffset];
    PetscInt  r          = edges->cell_ids[cellOffset + 1];
    PetscReal bl         = b_ptr[l];
    PetscReal br         = b_ptr[r];

    cn_vec_int[ii] = edges->cn[iedge];
    sn_vec_int[ii] = edges->sn[iedge];

    if (bl == 1 && br == 0) {
      // Update left values as it is a reflective boundary wall
      hl_vec_int[ii] = hr_vec_int[ii];

      PetscReal dum1 = Square(sn_vec_int[ii]) - Square(cn_vec_int[ii]);
      PetscReal dum2 = 2.0 * sn_vec_int[ii] * cn_vec_int[ii];

      ul_vec_int[ii] = ur_vec_int[ii] * dum1 - vr_vec_int[ii] * dum2;
      vl_vec_int[ii] = -ur_vec_int[ii] * dum2 - vr_vec_int[ii] * dum1;

    } else if (bl == 0 && br == 1) {
      // Update right values as it is a reflective boundary wall
      hr_vec_int[ii] = hl_vec_int[ii];

      PetscReal dum1 = Square(sn_vec_int[ii]) - Square(cn_vec_int[ii]);
      PetscReal dum2 = 2.0 * sn_vec_int[ii] * cn_vec_int[ii];

      ur_vec_int[ii] = ul_vec_int[ii] * dum1 - vl_vec_int[ii] * dum2;
      vr_vec_int[ii] = -ul_vec_int[ii] * dum2 - vl_vec_int[ii] * dum1;
    }
  }

  // Call Riemann solver
  PetscCall(solver(num, hl_vec_int, hr_vec_int, ul_vec_int, ur_vec_int, vl_vec_int, vr_vec_int, sn_vec_int, cn_vec_int, flux_vec_int, amax_vec_int));

  // Save the flux values in the Vec based by TS
  for (PetscInt ii = 0; ii < mesh->num_internal_edges; ii++) {
    PetscInt  iedge      = edges->internal_edge_ids[ii];
    PetscInt  cellOffset = edges->cell_offsets[iedge];
    PetscInt  l          = edges->cell_ids[cellOffset];
    PetscInt  r          = edges->cell_ids[cellOffset + 1];
    PetscReal edgeLen    = edges->lengths[iedge];

    PetscReal hl = x_ptr[l * ndof + 0];
    PetscReal hr = x_ptr[r * ndof + 0];
    PetscReal bl = b_ptr[l];
    PetscReal br = b_ptr[r];

    *amax_value = fmax(*amax_value, amax_vec_int[ii]);

    if (bl == 0 && br == 0) {
      // Both, left and right cells are not reflective boundary walls
      if (!(hr < app->tiny_h && hl < app->tiny_h)) {
        PetscReal areal = cells->areas[l];
        PetscReal arear = cells->areas[r];

        for (PetscInt idof = 0; idof < ndof; idof++) {
          if (cells->is_local[l]) f_ptr[l * ndof + idof] -= flux_vec_int[ii][idof] * edgeLen / areal;
          if (cells->is_local[r]) f_ptr[r * ndof + idof] += flux_vec_int[ii][idof] * edgeLen / arear;
        }
      }

    } else if (bl == 1 && br == 0) {
      // Left cell is a reflective boundary wall and right cell is an internal cell

      PetscReal arear = cells->areas[r];
      for (PetscInt idof = 0; idof < ndof; idof++) {
        if (cells->is_local[r]) f_ptr[r * ndof + idof] += flux_vec_int[ii][idof] * edgeLen / arear;
      }

    } else if (bl == 0 && br == 1) {
      // Left cell is an internal cell and right cell is a reflective boundary wall

      PetscReal areal = cells->areas[l];
      for (PetscInt idof = 0; idof < ndof; idof++) {
        if (cells->is_local[l]) f_ptr[l * ndof + idof] -= flux_vec_int[ii][idof] * edgeLen / areal;
      }
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(app->localX, &x_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));
  PetscCall(VecRestoreArray(app->localB, &b_ptr));

  PetscFunctionReturn(0);
}

/// @brief It computes RHSFunction for boundary edges
/// @param [inout] app A RDyApp struct
/// @param [inout] F A global flux Vec
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RHSFunctionForBoundaryEdges(RDyApp app, Vec F, PetscReal *amax_value) {
  PetscFunctionBeginUser;

  RDyMesh  *mesh  = &app->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  // Get pointers to vector data
  PetscScalar *x_ptr, *f_ptr, *b_ptr;
  PetscCall(VecGetArray(app->localX, &x_ptr));
  PetscCall(VecGetArray(F, &f_ptr));
  PetscCall(VecGetArray(app->localB, &b_ptr));

  PetscInt  ndof = app->ndof;
  PetscInt  num  = mesh->num_boundary_edges;
  PetscReal hl_vec_bnd[num], hul_vec_bnd[num], hvl_vec_bnd[num], ul_vec_bnd[num], vl_vec_bnd[num];
  PetscReal hr_vec_bnd[num], ur_vec_bnd[num], vr_vec_bnd[num];
  PetscReal sn_vec_bnd[num], cn_vec_bnd[num];
  PetscReal flux_vec_bnd[num][3], amax_vec_bnd[num];

  // Collect the h/hu/hv for left cells to compute u/v
  for (PetscInt ii = 0; ii < mesh->num_boundary_edges; ii++) {
    PetscInt iedge      = edges->boundary_edge_ids[ii];
    PetscInt cellOffset = edges->cell_offsets[iedge];
    PetscInt l          = edges->cell_ids[cellOffset];

    hl_vec_bnd[ii]  = x_ptr[l * ndof + 0];
    hul_vec_bnd[ii] = x_ptr[l * ndof + 1];
    hvl_vec_bnd[ii] = x_ptr[l * ndof + 2];
  }

  // Compute u/v for left cells
  PetscCall(GetVelocityFromMomentum(num, app->tiny_h, hl_vec_bnd, hul_vec_bnd, hvl_vec_bnd, ul_vec_bnd, vl_vec_bnd));

  // Compute h/u/v for right cells
  for (PetscInt ii = 0; ii < mesh->num_boundary_edges; ii++) {
    PetscInt iedge      = edges->boundary_edge_ids[ii];
    PetscInt cellOffset = edges->cell_offsets[iedge];
    PetscInt l          = edges->cell_ids[cellOffset];

    cn_vec_bnd[ii] = edges->cn[iedge];
    sn_vec_bnd[ii] = edges->sn[iedge];

    if (cells->is_local[l] && b_ptr[l] == 0) {
      // Perform computation for a boundary edge

      if (cells->is_local[l] && b_ptr[l] == 0) {
        hr_vec_bnd[ii] = hl_vec_bnd[ii];

        PetscReal dum1 = Square(sn_vec_bnd[ii]) - Square(cn_vec_bnd[ii]);
        PetscReal dum2 = 2.0 * sn_vec_bnd[ii] * cn_vec_bnd[ii];

        ur_vec_bnd[ii] = ul_vec_bnd[ii] * dum1 - vl_vec_bnd[ii] * dum2;
        vr_vec_bnd[ii] = -ul_vec_bnd[ii] * dum2 - vl_vec_bnd[ii] * dum1;
      }
    }
  }

  // Call Riemann solver
  PetscCall(solver(num, hl_vec_bnd, hr_vec_bnd, ul_vec_bnd, ur_vec_bnd, vl_vec_bnd, vr_vec_bnd, sn_vec_bnd, cn_vec_bnd, flux_vec_bnd, amax_vec_bnd));

  // Save the flux values in the Vec based by TS
  for (PetscInt ii = 0; ii < mesh->num_boundary_edges; ii++) {
    PetscInt  iedge      = edges->boundary_edge_ids[ii];
    PetscInt  cellOffset = edges->cell_offsets[iedge];
    PetscInt  l          = edges->cell_ids[cellOffset];
    PetscReal edgeLen    = edges->lengths[iedge];
    PetscReal areal      = cells->areas[l];

    if (cells->is_local[l] && b_ptr[l] == 0) {
      // Perform computation for a boundary edge

      PetscReal hl = x_ptr[l * ndof + 0];

      if (!(hl < app->tiny_h)) {
        *amax_value = fmax(*amax_value, amax_vec_bnd[ii]);
        for (PetscInt idof = 0; idof < ndof; idof++) {
          f_ptr[l * ndof + idof] -= flux_vec_bnd[ii][idof] * edgeLen / areal;
        }
      }
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(app->localX, &x_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));
  PetscCall(VecRestoreArray(app->localB, &b_ptr));

  PetscFunctionReturn(0);
}

/// @brief Add contribution of the source term of SWE
/// @param [in] app A RDyApp struct
/// @param [inout] F A global flux Vec
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode AddSourceTerm(RDyApp app, Vec F) {
  PetscFunctionBeginUser;

  RDyMesh  *mesh  = &app->mesh;
  RDyCells *cells = &mesh->cells;

  // Get access to Vec
  PetscScalar *x_ptr, *f_ptr;
  PetscCall(VecGetArray(app->localX, &x_ptr));
  PetscCall(VecGetArray(F, &f_ptr));

  PetscInt ndof = app->ndof;

  PetscInt  num = mesh->num_cells;
  PetscReal h_vec[num], hu_vec[num], hv_vec[num], u_vec[num], v_vec[num];

  // Collect the h/hu/hv for cells to compute u/v
  for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
    h_vec[icell]  = x_ptr[icell * ndof + 0];
    hu_vec[icell] = x_ptr[icell * ndof + 1];
    hv_vec[icell] = x_ptr[icell * ndof + 2];
  }

  // Compute u/v for cells
  PetscCall(GetVelocityFromMomentum(num, app->tiny_h, h_vec, hu_vec, hv_vec, u_vec, v_vec));

  for (PetscInt icell = 0; icell < mesh->num_cells; icell++) {
    if (cells->is_local[icell]) {
      PetscReal h  = h_vec[icell];
      PetscReal hu = hu_vec[icell];
      PetscReal hv = hv_vec[icell];

      PetscReal dz_dx = cells->dz_dx[icell];
      PetscReal dz_dy = cells->dz_dy[icell];

      PetscReal bedx = dz_dx * GRAVITY * h;
      PetscReal bedy = dz_dy * GRAVITY * h;

      PetscReal u = u_vec[icell];
      PetscReal v = u_vec[icell];

      PetscReal Fsum_x = f_ptr[icell * ndof + 1];
      PetscReal Fsum_y = f_ptr[icell * ndof + 2];

      PetscReal tbx = 0.0, tby = 0.0;

      if (h >= app->tiny_h) {
        // Manning's coefficient
        PetscReal Uniform_roughness = 0.015;
        PetscReal N_mannings        = GRAVITY * Uniform_roughness * Uniform_roughness;

        // Cd = g n^2 h^{-1/3}, where n is Manning's coefficient
        PetscReal Cd = GRAVITY * Square(N_mannings) * PetscPowReal(h, -1.0 / 3.0);

        PetscReal velocity = PetscSqrtReal(Square(u) + Square(v));

        PetscReal tb = Cd * velocity / h;

        PetscReal dt     = app->dt;
        PetscReal factor = tb / (1.0 + dt * tb);

        tbx = (hu + dt * Fsum_x - dt * bedx) * factor;
        tby = (hv + dt * Fsum_y - dt * bedy) * factor;
      }

      f_ptr[icell * ndof + 0] += 0.0;
      f_ptr[icell * ndof + 1] += -bedx - tbx;
      f_ptr[icell * ndof + 2] += -bedy - tby;
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(app->localX, &x_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));

  PetscFunctionReturn(0);
}

/// @brief It is the RHSFunction called by TS
/// @param [in] ts A TS struct
/// @param [in] t Time
/// @param [in] X A global solution Vec
/// @param [inout] F A global flux Vec
/// @param [inout] ptr A user-defined pointer
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ptr) {
  PetscFunctionBeginUser;

  RDyApp app = ptr;
  DM     dm  = app->dm;

  app->tstep = app->tstep + 1;

  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, app->localX));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, app->localX));
  PetscCall(VecZeroEntries(F));

  PetscReal amax_value = 0.0;
  PetscCall(RHSFunctionForInternalEdges(app, F, &amax_value));
  PetscCall(RHSFunctionForBoundaryEdges(app, F, &amax_value));
  PetscCall(AddSourceTerm(app, F));

  if (app->save) {
    char fname[PETSC_MAX_PATH_LEN];
    sprintf(fname, "outputs/ex2b_Nx_%d_Ny_%d_dt_%f_%d_np%d.dat", app->Nx, app->Ny, app->dt, app->tstep - 1, app->comm_size);
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(app->comm, fname, FILE_MODE_WRITE, &viewer));

    Vec natural;
    PetscCall(DMPlexCreateNaturalVector(app->dm, &natural));
    PetscCall(DMPlexGlobalToNaturalBegin(app->dm, X, natural));
    PetscCall(DMPlexGlobalToNaturalEnd(app->dm, X, natural));
    PetscCall(VecView(natural, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    sprintf(fname, "outputs/ex2b_flux_Nx_%d_Ny_%d_dt_%f_%d_np%d.dat", app->Nx, app->Ny, app->dt, app->tstep - 1, app->comm_size);
    PetscCall(PetscViewerBinaryOpen(app->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(DMPlexGlobalToNaturalBegin(app->dm, F, natural));
    PetscCall(DMPlexGlobalToNaturalEnd(app->dm, F, natural));
    PetscCall(VecView(natural, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(VecDestroy(&natural));
  }

  PetscPrintf(PETSC_COMM_SELF, "Time Step = %d, rank = %d, Courant Number = %f\n", 1, app->rank, amax_value * app->dt * 2);

  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  RDyApp app;
  PetscCall(PetscNew(&app));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Create DMs
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "1. Create DMs\n"));
  PetscCall(CreateDM(app));
  PetscCall(CreateAuxDM(app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Create vectors for solution and residual
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  Vec X, R;
  PetscCall(DMCreateGlobalVector(app->dm, &X));  // size = dof * number of cells
  PetscCall(VecDuplicate(X, &R));
  PetscCall(VecViewFromOptions(X, NULL, "-vec_view"));
  PetscCall(DMCreateLocalVector(app->dm, &app->localX));  // size = dof * number of cells
  PetscCall(DMCreateGlobalVector(app->auxdm, &app->B));
  PetscCall(DMCreateLocalVector(app->auxdm, &app->localB));  // size = dof * number of cells

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Create the mesh
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2. CreateMesh\n"));
  PetscCall(RDyMeshCreateFromDM(app->dm, &app->mesh));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Initial Condition
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "3. SetInitialCondition\n"));
  size_t len;

  PetscStrlen(app->initial_condition_filename, &len);
  if (!len) {
    PetscCall(SetInitialCondition(app, X));
  }
  else {
    PetscCall(SetInitialConditionFromFile(app, X));
  }
  
  {
    char fname[PETSC_MAX_PATH_LEN];
    sprintf(fname, "outputs/ex2b_Nx_%d_Ny_%d_dt_%f_IC_np%d.dat", app->Nx, app->Ny, app->dt, app->comm_size);

    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(app->comm, fname, FILE_MODE_WRITE, &viewer));
    Vec natural;
    PetscCall(DMPlexCreateNaturalVector(app->dm, &natural));
    PetscCall(DMPlexGlobalToNaturalBegin(app->dm, X, natural));
    PetscCall(DMPlexGlobalToNaturalEnd(app->dm, X, natural));
    PetscCall(VecView(natural, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(VecDestroy(&natural));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Add buildings
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  if (app->add_building) {
    PetscCall(AddBuildings(app));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Create timestepping solver context
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscReal max_time = app->Nt * app->dt;
  TS        ts;
  PetscCall(TSCreate(app->comm, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSEULER));
  PetscCall(TSSetDM(ts, app->dm));
  PetscCall(TSSetRHSFunction(ts, R, RHSFunction, app));
  PetscCall(TSSetMaxTime(ts, max_time));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(TSSetTimeStep(ts, app->dt));

  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSolve(ts, X));

  if (app->save) {
    char fname[PETSC_MAX_PATH_LEN];
    sprintf(fname, "outputs/ex2b_Nx_%d_Ny_%d_dt_%f_%d_np%d.dat", app->Nx, app->Ny, app->dt, app->Nt, app->comm_size);

    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(app->comm, fname, FILE_MODE_WRITE, &viewer));
    Vec natural;
    PetscCall(DMPlexCreateNaturalVector(app->dm, &natural));
    PetscCall(DMPlexGlobalToNaturalBegin(app->dm, X, natural));
    PetscCall(DMPlexGlobalToNaturalEnd(app->dm, X, natural));
    PetscCall(VecView(natural, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(VecDestroy(&natural));
  }

  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&R));
  PetscCall(VecDestroy(&app->B));
  PetscCall(VecDestroy(&app->localB));
  PetscCall(VecDestroy(&app->localX));
  PetscCall(VecDestroy(&R));
  PetscCall(RDyMeshDestroy(app->mesh));
  PetscCall(DMDestroy(&app->dm));
  PetscCall(RDyFree(app));

  PetscCall(PetscFinalize());

  return 0;
}
