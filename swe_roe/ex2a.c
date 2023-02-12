static char help[] = "Partial 2D dam break problem.\n";

#include <assert.h>
#include <math.h>
#include <petsc.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <petscts.h>
#include <petscvec.h>

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

  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscInt  icell = c - cStart;
    PetscInt  gref, junkInt;
    PetscInt  dim = 2;
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

    PetscInt coordOffset, dim = 2;
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

  /// PETSC_TRUE iff edge is shared by locally owned cells, OR
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

  for (PetscInt e = eStart; e < eEnd; e++) {
    PetscInt  iedge = e - eStart;
    PetscInt  dim   = 2;
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

/// Creates a fully initialized RDyEdges struct from a given DM.
/// @param [in] dm A DM that provides edge data
/// @param [out] edges A pointer to an RDyEdges that stores allocated data.
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode RDyComputeAdditionalEdgeGeoAttributes(DM dm, RDyCells *cells, RDyEdges *edges) {
  PetscFunctionBegin;

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

    PetscBool is_internal_edge = (r >= 0 && l >= 0);

    if (PetscAbs(edges->normals[iedge].V[0]) < 1.e-10) {
      // It is a vertical edge, so
      // cn = 0.0 and sn = +/- 1.0

      if (is_internal_edge) {
        PetscReal yr     = cells->centroids[r].X[1];
        PetscReal yl     = cells->centroids[l].X[1];
        PetscReal dy_l2r = yr - yl;
        if (dy_l2r < 0.0) {
          edges->sn[iedge] = -1.0;
        } else {
          edges->sn[iedge] = 1.0;
        }
      } else {
        edges->sn[iedge] = 1.0;
      }

    } else if (PetscAbs(edges->normals[iedge].V[1]) < 1.e-10) {
      // It is a horizontal edge, so
      // sn = 0.0 and cn = +/- 1.0
      if (is_internal_edge) {
        PetscReal xr     = cells->centroids[r].X[0];
        PetscReal xl     = cells->centroids[l].X[0];
        PetscReal dx_l2r = xr - xl;
        if (dx_l2r < 0.0) {
          edges->cn[iedge] = -1.0;
        } else {
          edges->cn[iedge] = 1.0;
        }
      } else {
        edges->cn[iedge] = 1.0;
      }

    } else {
      printf("The code only support quad cells with edges that align with x and y axis\n");
      exit(0);
    }
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
  mesh->num_edges = eEnd - eStart;

  // Determine the number of vertices in the mesh
  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);
  mesh->num_vertices = vEnd - vStart;

  // Create mesh elements from the DM
  PetscCall(RDyCellsCreateFromDM(dm, &mesh->cells));
  PetscCall(RDyEdgesCreateFromDM(dm, &mesh->edges));
  PetscCall(RDyVerticesCreateFromDM(dm, &mesh->vertices));
  PetscCall(RDyComputeAdditionalEdgeGeoAttributes(dm, &mesh->cells, &mesh->edges));

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

  PetscInt  dof;
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
  app->dof    = 3;

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

    PetscCall(DMPlexCreateBoxMesh(app->comm, dim, PETSC_FALSE, faces, lower, upper, PETSC_NULLPTR, PETSC_TRUE, &app->dm));
  } else {
    DMPlexCreateFromFile(app->comm, app->filename, "ex2.c", PETSC_FALSE, &app->dm);
  }
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
    PetscInt ndof = 3;
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
PetscErrorCode solver(PetscReal hl, PetscReal hr, PetscReal ul, PetscReal ur, PetscReal vl, PetscReal vr, PetscReal sn, PetscReal cn,
                      PetscScalar *fij, PetscScalar *amax) {
  PetscFunctionBeginUser;

  PetscReal grav = 9.806;

  // Compute Roe averages
  PetscReal duml  = pow(hl, 0.5);
  PetscReal dumr  = pow(hr, 0.5);
  PetscReal cl    = pow(grav * hl, 0.5);
  PetscReal cr    = pow(grav * hr, 0.5);
  PetscReal hhat  = duml * dumr;
  PetscReal uhat  = (duml * ul + dumr * ur) / (duml + dumr);
  PetscReal vhat  = (duml * vl + dumr * vr) / (duml + dumr);
  PetscReal chat  = pow(0.5 * grav * (hl + hr), 0.5);
  PetscReal uperp = uhat * cn + vhat * sn;

  PetscReal dh     = hr - hl;
  PetscReal du     = ur - ul;
  PetscReal dv     = vr - vl;
  PetscReal dupar  = -du * sn + dv * cn;
  PetscReal duperp = du * cn + dv * sn;

  PetscReal dW[3];
  dW[0] = 0.5 * (dh - hhat * duperp / chat);
  dW[1] = hhat * dupar;
  dW[2] = 0.5 * (dh + hhat * duperp / chat);

  PetscReal uperpl = ul * cn + vl * sn;
  PetscReal uperpr = ur * cn + vr * sn;
  PetscReal al1    = uperpl - cl;
  PetscReal al3    = uperpl + cl;
  PetscReal ar1    = uperpr - cr;
  PetscReal ar3    = uperpr + cr;

  PetscReal R[3][3];
  R[0][0] = 1.0;
  R[0][1] = 0.0;
  R[0][2] = 1.0;
  R[1][0] = uhat - chat * cn;
  R[1][1] = -sn;
  R[1][2] = uhat + chat * cn;
  R[2][0] = vhat - chat * sn;
  R[2][1] = cn;
  R[2][2] = vhat + chat * sn;

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
  FL[0] = uperpl * hl;
  FL[1] = ul * uperpl * hl + 0.5 * grav * hl * hl * cn;
  FL[2] = vl * uperpl * hl + 0.5 * grav * hl * hl * sn;

  FR[0] = uperpr * hr;
  FR[1] = ur * uperpr * hr + 0.5 * grav * hr * hr * cn;
  FR[2] = vr * uperpr * hr + 0.5 * grav * hr * hr * sn;

  // fij = 0.5*(FL + FR - matmul(R,matmul(A,dW))
  fij[0] = 0.5 * (FL[0] + FR[0] - R[0][0] * A[0][0] * dW[0] - R[0][1] * A[1][1] * dW[1] - R[0][2] * A[2][2] * dW[2]);
  fij[1] = 0.5 * (FL[1] + FR[1] - R[1][0] * A[0][0] * dW[0] - R[1][1] * A[1][1] * dW[1] - R[1][2] * A[2][2] * dW[2]);
  fij[2] = 0.5 * (FL[2] + FR[2] - R[2][0] * A[0][0] * dW[0] - R[2][1] * A[1][1] * dW[1] - R[2][2] * A[2][2] * dW[2]);

  *amax = chat + fabs(uperp);

  PetscFunctionReturn(0);
}

/// @brief Computes velocities in x and y-dir based on momentum in x and y-dir
/// @param tiny_h Threshold value for height
/// @param h Height
/// @param hu Momentum in x-dir
/// @param hv Momentum in y-dir
/// @param u Velocity in x-dir
/// @param v Velocit in y-dir
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode GetVelocityFromMomentum(PetscReal tiny_h, PetscReal h, PetscReal hu, PetscReal hv, PetscReal *u, PetscReal *v) {
  PetscFunctionBeginUser;

  if (h < tiny_h) {
    *u = 0.0;
    *v = 0.0;
  } else {
    *u = hu / h;
    *v = hv / h;
  }

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

  DM        dm    = app->dm;
  RDyMesh  *mesh  = &app->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;
  // RDyVertices *vertices = &mesh->vertices;

  app->tstep = app->tstep + 1;

  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, app->localX));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, app->localX));
  PetscCall(VecZeroEntries(F));

  // Get pointers to vector data
  PetscScalar *x_ptr, *f_ptr, *b_ptr;
  PetscCall(VecGetArray(app->localX, &x_ptr));
  PetscCall(VecGetArray(F, &f_ptr));
  PetscCall(VecGetArray(app->localB, &b_ptr));

  PetscInt  dof        = 3;
  PetscReal amax_value = 0.0;

  for (PetscInt iedge = 0; iedge < mesh->num_edges; iedge++) {
    PetscInt  cellOffset = edges->cell_offsets[iedge];
    PetscInt  l          = edges->cell_ids[cellOffset];
    PetscInt  r          = edges->cell_ids[cellOffset + 1];
    PetscReal edgeLen    = edges->lengths[iedge];
    PetscReal cn         = edges->cn[iedge];
    PetscReal sn         = edges->sn[iedge];

    PetscBool is_edge_vertical;
    if (PetscAbs(edges->normals[iedge].V[0]) < 1.e-10) {
      is_edge_vertical = PETSC_TRUE;
    } else if (PetscAbs(edges->normals[iedge].V[1]) < 1.e-10) {
      is_edge_vertical = PETSC_FALSE;
    } else {
      printf("The code only support quad cells with edges that align with x and y axis\n");
      exit(0);
    }

    if (r >= 0 && l >= 0) {
      // Perform computation for an internal edge

      PetscReal hl = x_ptr[l * dof + 0];
      PetscReal hr = x_ptr[r * dof + 0];
      PetscReal bl = b_ptr[l];
      PetscReal br = b_ptr[r];

      if (bl == 0 && br == 0) {
        // Both, left and right cells are not boundary walls
        if (!(hr < app->tiny_h && hl < app->tiny_h)) {
          PetscReal hul   = x_ptr[l * dof + 1];
          PetscReal hvl   = x_ptr[l * dof + 2];
          PetscReal hur   = x_ptr[r * dof + 1];
          PetscReal hvr   = x_ptr[r * dof + 2];
          PetscReal areal = cells->areas[l];
          PetscReal arear = cells->areas[r];

          PetscReal ur, vr, ul, vl;

          PetscCall(GetVelocityFromMomentum(app->tiny_h, hr, hur, hvr, &ur, &vr));
          PetscCall(GetVelocityFromMomentum(app->tiny_h, hl, hul, hvl, &ul, &vl));

          PetscReal flux[3], amax;
          PetscCall(solver(hl, hr, ul, ur, vl, vr, sn, cn, flux, &amax));
          amax_value = fmax(amax_value, amax);

          for (PetscInt idof = 0; idof < dof; idof++) {
            if (cells->is_local[l]) f_ptr[l * dof + idof] -= flux[idof] * edgeLen / areal;
            if (cells->is_local[r]) f_ptr[r * dof + idof] += flux[idof] * edgeLen / arear;
          }
        }

      } else if (bl == 1 && br == 0) {
        // Left cell is a boundary wall and right cell is an internal cell

        PetscReal hr  = x_ptr[r * dof + 0];
        PetscReal hur = x_ptr[r * dof + 1];
        PetscReal hvr = x_ptr[r * dof + 2];

        PetscReal ur, vr;
        PetscCall(GetVelocityFromMomentum(app->tiny_h, hr, hur, hvr, &ur, &vr));

        PetscReal hl = hr;
        PetscReal ul, vl;
        if (is_edge_vertical) {
          ul = ur;
          vl = -vr;
        } else {
          ul = -ur;
          vl = vr;
        }

        PetscReal flux[3], amax;
        PetscCall(solver(hl, hr, ul, ur, vl, vr, sn, cn, flux, &amax));
        amax_value = fmax(amax_value, amax);

        PetscReal arear = cells->areas[r];
        for (PetscInt idof = 0; idof < dof; idof++) {
          if (cells->is_local[r]) f_ptr[r * dof + idof] += flux[idof] * edgeLen / arear;
        }

      } else if (bl == 0 && br == 1) {
        // Left cell is an internal cell and right cell is a boundary wall

        PetscReal hl  = x_ptr[l * dof + 0];
        PetscReal hul = x_ptr[l * dof + 1];
        PetscReal hvl = x_ptr[l * dof + 2];

        PetscReal ul, vl;
        PetscCall(GetVelocityFromMomentum(app->tiny_h, hl, hul, hvl, &ul, &vl));

        PetscReal hr = hl;
        PetscReal ur, vr;
        if (is_edge_vertical) {
          ur = ul;
          vr = -vl;
        } else {
          ur = -ul;
          vr = vl;
        }

        PetscReal flux[3], amax;
        PetscCall(solver(hl, hr, ul, ur, vl, vr, sn, cn, flux, &amax));
        amax_value = fmax(amax_value, amax);

        PetscReal areal = cells->areas[l];
        for (PetscInt idof = 0; idof < dof; idof++) {
          if (cells->is_local[l]) f_ptr[l * dof + idof] -= flux[idof] * edgeLen / areal;
        }
      }

    } else if (cells->is_local[l] && b_ptr[l] == 0) {
      // Perform computation for a boundary edge

      PetscBool bnd_cell_order_flipped = PETSC_FALSE;

      if (is_edge_vertical) {
        if (cells->centroids[l].X[1] > edges->centroids[iedge].X[1]) bnd_cell_order_flipped = PETSC_TRUE;
      } else {
        if (cells->centroids[l].X[0] > edges->centroids[iedge].X[0]) bnd_cell_order_flipped = PETSC_TRUE;
      }

      PetscReal hl = x_ptr[l * dof + 0];

      if (!(hl < app->tiny_h)) {
        PetscReal hul = x_ptr[l * dof + 1];
        PetscReal hvl = x_ptr[l * dof + 2];

        PetscReal ul, vl;
        PetscCall(GetVelocityFromMomentum(app->tiny_h, hl, hul, hvl, &ul, &vl));

        PetscReal hr, ur, vr;
        hr = hl;
        if (is_edge_vertical) {
          ur = ul;
          vr = -vl;
        } else {
          ur = -ul;
          vr = vl;
        }

        if (bnd_cell_order_flipped) {
          PetscReal tmp;
          tmp = hl;
          hl  = hr;
          hr  = tmp;
          tmp = ul;
          ul  = ur;
          ur  = tmp;
          tmp = vl;
          vl  = vr;
          vr  = tmp;
        }

        PetscReal flux[3], amax;
        PetscCall(solver(hl, hr, ul, ur, vl, vr, sn, cn, flux, &amax));
        amax_value = fmax(amax_value, amax);

        PetscReal areal = cells->areas[l];
        for (PetscInt idof = 0; idof < dof; idof++) {
          if (!bnd_cell_order_flipped) {
            f_ptr[l * dof + idof] -= flux[idof] * edgeLen / areal;
          } else {
            f_ptr[l * dof + idof] += flux[idof] * edgeLen / areal;
          }
        }
      }
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(app->localX, &x_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));
  PetscCall(VecRestoreArray(app->localB, &b_ptr));

  if (app->save) {
    char fname[PETSC_MAX_PATH_LEN];
    sprintf(fname, "outputs/ex2a_Nx_%d_Ny_%d_dt_%f_%d_np%d.dat", app->Nx, app->Ny, app->dt, app->tstep - 1, app->comm_size);
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(app->comm, fname, FILE_MODE_WRITE, &viewer));

    Vec natural;
    PetscCall(DMPlexCreateNaturalVector(app->dm, &natural));
    PetscCall(DMPlexGlobalToNaturalBegin(app->dm, X, natural));
    PetscCall(DMPlexGlobalToNaturalEnd(app->dm, X, natural));
    PetscCall(VecView(natural, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    sprintf(fname, "outputs/ex2a_flux_Nx_%d_Ny_%d_dt_%f_%d_np%d.dat", app->Nx, app->Ny, app->dt, app->tstep - 1, app->comm_size);
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
  PetscCall(SetInitialCondition(app, X));
  {
    char fname[PETSC_MAX_PATH_LEN];
    sprintf(fname, "outputs/ex2a_Nx_%d_Ny_%d_dt_%f_IC_np%d.dat", app->Nx, app->Ny, app->dt, app->comm_size);

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
    sprintf(fname, "outputs/ex2a_Nx_%d_Ny_%d_dt_%f_%d_np%d.dat", app->Nx, app->Ny, app->dt, app->Nt, app->comm_size);

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
