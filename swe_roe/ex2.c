static char help[] = "Partial 2D dam break problem.\n";

#include <assert.h>
#include <math.h>
#include <petsc.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <petscts.h>
#include <petscvec.h>

#define RDyAlloc(size, result) PetscCalloc1(size, result)

typedef struct _n_User *User;

typedef struct {
  PetscReal X[3];
} RDyCoordinate;

typedef struct {
  PetscReal V[3];
} RDyVector;

typedef enum {
  CELL_TRI_TYPE = 0, /* tetrahedron cell for a 3D cell */
  CELL_QUAD_TYPE     /* hexahedron cell for a 3D cell */
} RDyCellType;

typedef struct {
  PetscInt *id;         /* id of the cell in local numbering */
  PetscInt *global_id;  /* global id of the cell in local numbering */
  PetscInt *natural_id; /* natural id of the cell in local numbering */

  PetscBool *is_local;

  PetscInt *num_vertices;  /* number of vertices of the cell    */
  PetscInt *num_edges;     /* number of edges of the cell       */
  PetscInt *num_neighbors; /* number of neigbors of the cell    */

  PetscInt *vertex_offset;   /* vertice IDs that form the cell    */
  PetscInt *edge_offset;     /* vertice IDs that form the cell    */
  PetscInt *neighbor_offset; /* vertice IDs that form the cell    */

  PetscInt *vertex_ids;   /* vertice IDs that form the cell    */
  PetscInt *edge_ids;     /* edge IDs that form the cell       */
  PetscInt *neighbor_ids; /* neighbor IDs that form the cell   */

  RDyCoordinate *centroid; /* cell centroid                     */

  PetscReal *area; /* area of the cell                */

} RDyCell;

typedef struct {
  PetscInt *id;        /* id of the vertex in local numbering                  */
  PetscInt *global_id; /* global id of the vertex in local numbering */

  PetscBool *is_local; /* true if the vertex is shared by a local cell         */

  PetscInt *num_cells; /* number of cells sharing the vertex          */
  PetscInt *num_edges; /* number of edges sharing the vertex                   */

  PetscInt *edge_offset; /* offset for edge IDs that share the vertex                       */
  PetscInt *cell_offset; /* offset for internal cell IDs that share the vertex              */

  PetscInt *edge_ids; /* edge IDs that share the vertex                       */
  PetscInt *cell_ids; /* internal cell IDs that share the vertex              */

  RDyCoordinate *coordinate; /* (x,y,z) location of the vertex                       */
} RDyVertex;

typedef struct {
  PetscInt *id;        /* id of the edge in local numbering         */
  PetscInt *global_id; /* global id of the edge in local numbering */

  PetscBool *is_local; /* true if the edge : (1) */
                       /* 1. Is shared by locally owned cells, or   */
                       /* 2. Is shared by local cell and non-local  */
                       /*    cell such that global ID of local cell */
                       /*    is smaller than the global ID of       */
                       /*    non-local cell */

  PetscInt *num_cells;  /* number of faces that form the edge        */
  PetscInt *vertex_ids; /* ids of vertices that form the edge        */

  PetscInt *cell_offset; /* offset for ids of cell that share the edge */
  PetscInt *cell_ids;    /* ids of cells that share the edge          */

  PetscBool *is_internal; /* false if the edge is on the mesh boundary */

  RDyVector     *normal;   /* unit normal vector                        */
  RDyCoordinate *centroid; /* edge centroid                             */

  PetscReal *length; /* length of the edge                        */

} RDyEdge;

typedef struct RDyMesh {
  PetscInt dim;

  PetscInt num_cells;
  PetscInt num_cells_local;
  PetscInt num_edges;
  PetscInt num_vertices;
  PetscInt num_boundary_faces;

  PetscInt max_vertex_cells, max_vertex_faces;

  RDyCell   cells;
  RDyVertex vertices;
  RDyEdge   edges;

  PetscInt *closureSize, **closure, maxClosureSize;

  PetscInt *nG2A;  // Mapping of global cells to application/natural cells

} RDyMesh;

struct _n_User {
  MPI_Comm  comm;
  char      filename[PETSC_MAX_PATH_LEN];
  DM        dm;
  PetscInt  Nt, Nx, Ny;
  PetscReal dt, dx, dy;
  PetscReal Lx, Ly;
  PetscReal hu, hd;
  PetscReal tiny_h;
  PetscInt  dof, rank, comm_size;
  Vec       B, localB;
  Vec       localX;
  PetscBool debug, save, add_building;
  PetscInt  tstep;
  PetscBool interpolate;

  RDyMesh *mesh;
};

PetscErrorCode RDyInitialize_IntegerArray_1D(PetscInt *array_1D, PetscInt ndim_1, PetscInt init_value) {
  PetscFunctionBegin;
  for (PetscInt i = 0; i < ndim_1; i++) {
    array_1D[i] = init_value;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RDyAllocate_IntegerArray_1D(PetscInt **array_1D, PetscInt ndim_1) {
  PetscFunctionBegin;
  PetscCall(RDyAlloc(ndim_1 * sizeof(PetscInt), array_1D));
  PetscCall(RDyInitialize_IntegerArray_1D(*array_1D, ndim_1, -1));
  PetscFunctionReturn(0);
}

PetscErrorCode RDyAllocate_RDyVector_1D(PetscInt ndim_1, RDyVector **array_1D) { return RDyAlloc(ndim_1 * sizeof(RDyVector), array_1D); }

PetscErrorCode RDyAllocate_RDyCoordinate_1D(PetscInt ndim_1, RDyCoordinate **array_1D) { return RDyAlloc(ndim_1 * sizeof(RDyCoordinate), array_1D); }

PetscErrorCode RDyAllocate_RealArray_1D(PetscReal **array_1D, PetscInt ndim_1) { return RDyAlloc(ndim_1 * sizeof(PetscReal), array_1D); }

PetscBool IsClosureWithinBounds(PetscInt closure, PetscInt start, PetscInt end) { return (closure >= start) && (closure < end); }

/// @brief Process command line options
/// @param [in] comm A MPI commmunicator
/// @param [inout] user A User data structure that is updated
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ProcessOptions(MPI_Comm comm, User user) {
  PetscFunctionBegin;

  user->comm   = comm;
  user->Nx     = 4;
  user->Ny     = 5;
  user->dx     = 1.0;
  user->dy     = 1.0;
  user->hu     = 10.0;  // water depth for the upstream of dam   [m]
  user->hd     = 5.0;   // water depth for the downstream of dam [m]
  user->tiny_h = 1e-7;
  user->dof    = 3;

  MPI_Comm_size(user->comm, &user->comm_size);
  MPI_Comm_rank(user->comm, &user->rank);

  PetscErrorCode ierr;
  ierr = PetscOptionsBegin(user->comm, NULL, "2D Mesh Options", "");
  PetscCall(ierr);
  {
    PetscCall(PetscOptionsInt("-Nx", "Number of cells in X", "", user->Nx, &user->Nx, NULL));
    PetscCall(PetscOptionsInt("-Ny", "Number of cells in Y", "", user->Ny, &user->Ny, NULL));
    PetscCall(PetscOptionsInt("-Nt", "Number of time steps", "", user->Nt, &user->Nt, NULL));
    PetscCall(PetscOptionsReal("-dx", "dx", "", user->dx, &user->dx, NULL));
    PetscCall(PetscOptionsReal("-dy", "dy", "", user->dy, &user->dy, NULL));
    PetscCall(PetscOptionsReal("-hu", "hu", "", user->hu, &user->hu, NULL));
    PetscCall(PetscOptionsReal("-hd", "hd", "", user->hd, &user->hd, NULL));
    PetscCall(PetscOptionsReal("-dt", "dt", "", user->dt, &user->dt, NULL));
    PetscCall(PetscOptionsBool("-b", "Add buildings", "", user->add_building, &user->add_building, NULL));
    PetscCall(PetscOptionsBool("-debug", "debug", "", user->debug, &user->debug, NULL));
    PetscCall(PetscOptionsBool("-save", "save outputs", "", user->save, &user->save, NULL));
    PetscCall(PetscOptionsString("-mesh_filename", "The mesh file", "ex2.c", user->filename, user->filename, PETSC_MAX_PATH_LEN, NULL));
  }
  ierr = PetscOptionsEnd();

  PetscCall(ierr);
  assert(user->hu >= 0.);
  assert(user->hd >= 0.);

  user->Lx = user->Nx * 1.0;
  user->Ly = user->Ny * 1.0;

  PetscReal max_time = user->Nt * user->dt;
  if (!user->rank) {
    PetscPrintf(user->comm, "Max simulation time is %f\n", max_time);
  }

  PetscFunctionReturn(0);
}

/// Creates the PETSc DM as a box or from a file. Add three DOFs and distribute the DM
/// @param [inout] user A User data structure that is modified
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateDM(User user) {
  PetscFunctionBegin;

  size_t len;

  PetscStrlen(user->filename, &len);
  if (!len) {
    PetscInt  dim     = 2;
    PetscInt  faces[] = {user->Nx, user->Ny};
    PetscReal lower[] = {0.0, 0.0};
    PetscReal upper[] = {user->Lx, user->Ly};

    PetscCall(DMPlexCreateBoxMesh(user->comm, dim, PETSC_FALSE, faces, lower, upper, PETSC_NULL, PETSC_TRUE, &user->dm));
  } else {
    DMPlexCreateFromFile(user->comm, user->filename, "ex2.c", PETSC_FALSE, &user->dm);
  }

  PetscObjectSetName((PetscObject)user->dm, "Mesh");
  PetscCall(DMSetFromOptions(user->dm));

  // Determine the number of cells, edges, and vertices of the mesh
  PetscInt cStart, cEnd;
  DMPlexGetHeightStratum(user->dm, 0, &cStart, &cEnd);

  // Create a single section that has 3 DOFs
  PetscSection sec;
  PetscCall(PetscSectionCreate(user->comm, &sec));

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
  PetscCall(DMSetLocalSection(user->dm, sec));
  PetscCall(PetscSectionViewFromOptions(sec, NULL, "-layout_view"));
  PetscCall(PetscSectionDestroy(&sec));
  PetscCall(DMSetBasicAdjacency(user->dm, PETSC_TRUE, PETSC_TRUE));

  // Before distributing the DM, set a flag to create mapping from natural-to-local order
  PetscCall(DMSetUseNatural(user->dm, PETSC_TRUE));

  // Distrubte the DM
  DM dmDist;
  PetscCall(DMPlexDistribute(user->dm, 1, NULL, &dmDist));
  if (dmDist) {
    DMDestroy(&user->dm);
    user->dm = dmDist;
  }
  PetscCall(DMViewFromOptions(user->dm, NULL, "-dm_view"));

  PetscFunctionReturn(0);
}

/// Allocates memory and initialize a RDyCell struct
/// @param [in] num_cells Number of cells
/// @param [out] cells A RDyCell struct that is allocated
///
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode AllocateCells(PetscInt num_cells, RDyCell *cells) {
  PetscFunctionBegin;

  PetscInt num_vertices  = 4;
  PetscInt num_edges     = 4;
  PetscInt num_neighbors = 4;

  PetscCall(RDyAllocate_IntegerArray_1D(&cells->id, num_cells));
  PetscCall(RDyAllocate_IntegerArray_1D(&cells->global_id, num_cells));
  PetscCall(RDyAllocate_IntegerArray_1D(&cells->natural_id, num_cells));

  PetscCall(RDyAlloc(num_cells * sizeof(PetscBool), &cells->is_local));

  PetscCall(RDyAllocate_IntegerArray_1D(&cells->num_vertices, num_cells));
  PetscCall(RDyAllocate_IntegerArray_1D(&cells->num_edges, num_cells));
  PetscCall(RDyAllocate_IntegerArray_1D(&cells->num_neighbors, num_cells));

  PetscCall(RDyAllocate_IntegerArray_1D(&cells->vertex_offset, num_cells + 1));
  PetscCall(RDyAllocate_IntegerArray_1D(&cells->edge_offset, num_cells + 1));
  PetscCall(RDyAllocate_IntegerArray_1D(&cells->neighbor_offset, num_cells + 1));

  PetscCall(RDyAllocate_IntegerArray_1D(&cells->vertex_ids, num_cells * num_vertices));
  PetscCall(RDyAllocate_IntegerArray_1D(&cells->edge_ids, num_cells * num_edges));
  PetscCall(RDyAllocate_IntegerArray_1D(&cells->neighbor_ids, num_cells * num_neighbors));

  PetscCall(RDyAllocate_RDyCoordinate_1D(num_cells, &cells->centroid));

  PetscCall(RDyAllocate_RealArray_1D(&cells->area, num_cells));

  for (PetscInt icell = 0; icell < num_cells; icell++) {
    cells->id[icell]            = icell;
    cells->num_vertices[icell]  = num_vertices;
    cells->num_edges[icell]     = num_edges;
    cells->num_neighbors[icell] = num_neighbors;
  }

  for (PetscInt icell = 0; icell <= num_cells; icell++) {
    cells->vertex_offset[icell]   = icell * num_vertices;
    cells->edge_offset[icell]     = icell * num_edges;
    cells->neighbor_offset[icell] = icell * num_neighbors;
  }

  PetscFunctionReturn(0);
}

/// Allocates memory and initialize a RDyEdge struct
/// @param [in] num_edges Number of edges
/// @param [out] edges A RDyEdge struct that is allocated
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode AllocateEdges(PetscInt num_edges, RDyEdge *edges) {
  PetscFunctionBegin;

  PetscInt num_cells = 2;

  PetscCall(RDyAllocate_IntegerArray_1D(&edges->id, num_edges));
  PetscCall(RDyAllocate_IntegerArray_1D(&edges->global_id, num_edges));
  PetscCall(RDyAllocate_IntegerArray_1D(&edges->num_cells, num_edges));
  PetscCall(RDyAllocate_IntegerArray_1D(&edges->vertex_ids, num_edges * 2));

  PetscCall(RDyAlloc(num_edges * sizeof(PetscBool), &edges->is_local));
  PetscCall(RDyAlloc(num_edges * sizeof(PetscBool), &edges->is_internal));

  PetscCall(RDyAllocate_IntegerArray_1D(&edges->cell_offset, num_edges + 1));
  PetscCall(RDyAllocate_IntegerArray_1D(&edges->cell_ids, num_edges * num_cells));

  PetscCall(RDyAllocate_RDyCoordinate_1D(num_edges, &edges->centroid));
  PetscCall(RDyAllocate_RDyVector_1D(num_edges, &edges->normal));

  PetscCall(RDyAllocate_RealArray_1D(&edges->length, num_edges));

  for (PetscInt iedge = 0; iedge < num_edges; iedge++) {
    edges->id[iedge]       = iedge;
    edges->is_local[iedge] = PETSC_FALSE;
  }
  for (PetscInt iedge = 0; iedge <= num_edges; iedge++) {
    edges->cell_offset[iedge] = iedge * num_cells;
  }

  PetscFunctionReturn(0);
}

/// Allocates memory and initialize a RDyVertex struct.
/// @param [in] num_vertices Number of vertices
/// @param [out] vertices A RDyVertex struct that is allocated
///
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode AllocateVertices(PetscInt num_vertices, RDyVertex *vertices) {
  PetscFunctionBegin;

  PetscInt ncells_per_vertex = 4;
  PetscInt nedges_per_vertex = 4;

  PetscCall(RDyAllocate_IntegerArray_1D(&vertices->id, num_vertices));
  PetscCall(RDyAllocate_IntegerArray_1D(&vertices->global_id, num_vertices));
  PetscCall(RDyAllocate_IntegerArray_1D(&vertices->num_cells, num_vertices));
  PetscCall(RDyAllocate_IntegerArray_1D(&vertices->num_edges, num_vertices));

  PetscCall(RDyAlloc(num_vertices * sizeof(PetscBool), &vertices->is_local));

  PetscCall(RDyAllocate_RDyCoordinate_1D(num_vertices, &vertices->coordinate));

  PetscCall(RDyAllocate_IntegerArray_1D(&vertices->edge_offset, num_vertices + 1));
  PetscCall(RDyAllocate_IntegerArray_1D(&vertices->cell_offset, num_vertices + 1));

  PetscCall(RDyAllocate_IntegerArray_1D(&vertices->edge_ids, num_vertices * nedges_per_vertex));
  PetscCall(RDyAllocate_IntegerArray_1D(&vertices->cell_ids, num_vertices * ncells_per_vertex));

  for (PetscInt ivertex = 0; ivertex < num_vertices; ivertex++) {
    vertices->id[ivertex]        = ivertex;
    vertices->is_local[ivertex]  = PETSC_FALSE;
    vertices->num_cells[ivertex] = 0;
    vertices->num_edges[ivertex] = 0;
  }

  for (PetscInt ivertex = 0; ivertex <= num_vertices; ivertex++) {
    vertices->edge_offset[ivertex] = ivertex * nedges_per_vertex;
    vertices->cell_offset[ivertex] = ivertex * ncells_per_vertex;
  }

  PetscFunctionReturn(0);
}

/// @brief Save cell-to-edge, cell-to-vertex, and cell geometric attributes.
/// @param [in] dm A PETSc DM
/// @param [inout] cells A RDyCell structure that is populated
/// @return
static PetscErrorCode PopulateCellsFromDM(DM dm, RDyCell *cells, PetscInt *num_cells_local) {
  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscInt eStart, eEnd;
  PetscInt vStart, vEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

  *num_cells_local = 0;

  for (PetscInt c = cStart; c < cEnd; c++) {
    PetscInt  icell = c - cStart;
    PetscInt  gref, junkInt;
    PetscInt  dim = 2;
    PetscReal centroid[dim], normal[dim];
    PetscCall(DMPlexGetPointGlobal(dm, c, &gref, &junkInt));
    DMPlexComputeCellGeometryFVM(dm, c, &cells->area[icell], &centroid[0], &normal[0]);

    for (PetscInt idim = 0; idim < dim; idim++) {
      cells->centroid[icell].X[idim] = centroid[idim];
    }

    PetscInt  pSize;
    PetscInt *p        = NULL;
    PetscInt  use_cone = PETSC_TRUE;

    cells->num_vertices[icell] = 0;
    cells->num_edges[icell]    = 0;
    if (gref >= 0) {
      cells->is_local[icell] = PETSC_TRUE;
      (*num_cells_local)++;
    } else {
      cells->is_local[icell] = PETSC_FALSE;
    }

    PetscCall(DMPlexGetTransitiveClosure(dm, c, use_cone, &pSize, &p));
    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], eStart, eEnd)) {
        PetscInt offset        = cells->edge_offset[icell];
        PetscInt index         = offset + cells->num_edges[icell];
        cells->edge_ids[index] = p[i] - eStart;
        cells->num_edges[icell]++;
      } else {
        PetscInt offset          = cells->vertex_offset[icell];
        PetscInt index           = offset + cells->num_vertices[icell];
        cells->vertex_ids[index] = p[i] - vStart;
        cells->num_vertices[icell]++;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, c, use_cone, &pSize, &p));
  }

  PetscFunctionReturn(0);
}

/// @brief Save edge-to-cell, edge-to-vertex, and geometric attributes.
/// @param [in] dm A PETSc DM
/// @param [inout] edges A RDyVertex structure that is populated
/// @return
static PetscErrorCode PopulateEdgesFromDM(DM dm, RDyEdge *edges) {
  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscInt eStart, eEnd;
  PetscInt vStart, vEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

  for (PetscInt e = eStart; e < eEnd; e++) {
    PetscInt  iedge = e - eStart;
    PetscInt  dim   = 2;
    PetscReal centroid[dim], normal[dim];
    DMPlexComputeCellGeometryFVM(dm, e, &edges->length[iedge], &centroid[0], &normal[0]);

    for (PetscInt idim = 0; idim < dim; idim++) {
      edges->centroid[iedge].X[idim] = centroid[idim];
      edges->normal[iedge].V[idim]   = normal[idim];
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
      PetscInt offset        = edges->cell_offset[iedge];
      PetscInt index         = offset + edges->num_cells[iedge];
      edges->cell_ids[index] = p[i] - cStart;
      edges->num_cells[iedge]++;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &pSize, &p));
  }

  PetscFunctionReturn(0);
}

/// @brief Save vertex-to-edge, vertex-to-cell, and geometric attributes
/// (e.g area).
/// @param [in] dm A PETSc DM
/// @param [inout] edges A RDyVertex structure that is populated
/// @return
static PetscErrorCode PopulateVerticesFromDM(DM dm, RDyVertex *vertices) {
  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscInt eStart, eEnd;
  PetscInt vStart, vEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

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
      vertices->coordinate[ivertex].X[idim] = coords[coordOffset + idim];
    }

    vertices->num_edges[ivertex] = 0;
    vertices->num_cells[ivertex] = 0;

    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], eStart, eEnd)) {
        PetscInt offset           = vertices->edge_offset[ivertex];
        PetscInt index            = offset + vertices->num_edges[ivertex];
        vertices->edge_ids[index] = p[i] - eStart;
        vertices->num_edges[ivertex]++;
      } else {
        PetscInt offset           = vertices->cell_offset[ivertex];
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

static PetscErrorCode SaveNaturalCellIDs(DM dm, RDyCell *cells, PetscInt rank) {
  PetscFunctionBegin;

  PetscBool useNatural;
  PetscCall(DMGetUseNatural(dm, &useNatural));

  if (useNatural) {
    PetscInt num_fields;

    PetscCall(DMGetNumFields(dm, &num_fields));

    // Create the natural vector
    Vec natural;
    PetscCall(DMCreateGlobalVector(dm, &natural));
    PetscInt natural_size = 0, cum_natural_size = 0;
    PetscCall(VecGetLocalSize(natural, &natural_size));
    PetscCall(MPI_Scan(&natural_size, &cum_natural_size, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD));

    // Add entries in the natural vector
    PetscScalar *entries;
    PetscCall(VecGetArray(natural, &entries));
    for (PetscInt i = 0; i < natural_size; ++i) {
      if (i % num_fields == 0) {
        entries[i] = i / num_fields + cum_natural_size / num_fields - natural_size / num_fields;
      } else {
        entries[i] = -1 - rank;
      }
    }
    PetscCall(VecRestoreArray(natural, &entries));
    VecView(natural, PETSC_VIEWER_STDOUT_WORLD);

    // Map natural IDs in global order
    Vec global;
    PetscCall(DMCreateGlobalVector(dm, &global));
    PetscCall(DMPlexNaturalToGlobalBegin(dm, natural, global));
    PetscCall(DMPlexNaturalToGlobalEnd(dm, natural, global));
    VecView(global, PETSC_VIEWER_STDOUT_WORLD);

    // Map natural IDs in local order
    Vec local;
    PetscCall(DMCreateLocalVector(dm, &local));
    PetscCall(DMGlobalToLocalBegin(dm, global, INSERT_VALUES, local));
    PetscCall(DMGlobalToLocalEnd(dm, global, INSERT_VALUES, local));
    VecView(local, PETSC_VIEWER_STDOUT_WORLD);

    // Save natural IDs
    PetscInt local_size;
    PetscCall(VecGetLocalSize(local, &local_size));
    PetscCall(VecGetArray(local, &entries));
    for (PetscInt i = 0; i < local_size / num_fields; ++i) {
      cells->natural_id[i] = entries[i * num_fields];
    }
    PetscCall(VecRestoreArray(local, &entries));

    // Cleanup
    PetscCall(VecDestroy(&natural));
    PetscCall(VecDestroy(&global));
    PetscCall(VecDestroy(&local));
  }

  PetscFunctionReturn(0);
}

/// Creates the RDyMesh structure from PETSc DM
/// @param [inout] user A User data structure that is modified
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateMesh(User user) {
  PetscFunctionBegin;

  PetscCall(RDyAlloc(sizeof(RDyMesh), &user->mesh));

  RDyMesh *mesh_ptr = user->mesh;

  // Determine the number of cells in the mesh
  PetscInt cStart, cEnd;
  DMPlexGetHeightStratum(user->dm, 0, &cStart, &cEnd);
  mesh_ptr->num_cells = cEnd - cStart;

  // Determine the number of edges in the mesh
  PetscInt eStart, eEnd;
  DMPlexGetDepthStratum(user->dm, 1, &eStart, &eEnd);
  mesh_ptr->num_edges = eEnd - eStart;

  // Determine the number of vertices in the mesh
  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(user->dm, 0, &vStart, &vEnd);
  mesh_ptr->num_vertices = vEnd - vStart;

  // Allocate memory for mesh elements
  PetscCall(AllocateCells(mesh_ptr->num_cells, &mesh_ptr->cells));
  PetscCall(AllocateEdges(mesh_ptr->num_edges, &mesh_ptr->edges));
  PetscCall(AllocateVertices(mesh_ptr->num_vertices, &mesh_ptr->vertices));

  // Populates mesh elements from PETSc DM
  PetscCall(PopulateCellsFromDM(user->dm, &mesh_ptr->cells, &mesh_ptr->num_cells_local));
  PetscCall(PopulateEdgesFromDM(user->dm, &mesh_ptr->edges));
  PetscCall(PopulateVerticesFromDM(user->dm, &mesh_ptr->vertices));
  // PetscCall(SaveNaturalCellIDs(user->dm, &mesh_ptr->cells, user->rank));

  PetscFunctionReturn(0);
}

/// @brief Sets initial condition for [h, hu, hv]
/// @param [in] user A User data structure
/// @param [inout] X Vec for initial condition
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode SetInitialCondition(User user, Vec X) {
  PetscFunctionBegin;

  RDyMesh *mesh  = user->mesh;
  RDyCell *cells = &mesh->cells;

  PetscCall(VecZeroEntries(X));

  PetscScalar *x_ptr;
  VecGetArray(X, &x_ptr);

  for (PetscInt icell = 0; icell < mesh->num_cells_local; icell++) {
    PetscInt ndof = 3;
    PetscInt idx  = icell * ndof;
    if (cells->centroid[icell].X[1] < 95.0) {
      x_ptr[idx] = user->hu;
    } else {
      x_ptr[idx] = user->hd;
    }
  }

  VecRestoreArray(X, &x_ptr);

  PetscFunctionReturn(0);
}

PetscErrorCode AddBuildings(User user) {
  PetscFunctionBeginUser;

  PetscInt bu = 30 / user->dx;
  PetscInt bd = 105 / user->dx;
  PetscInt bl = 95 / user->dy;
  PetscInt br = 105 / user->dy;

  PetscCall(VecZeroEntries(user->B));

  PetscScalar *b_ptr;
  PetscCall(VecGetArray(user->B, &b_ptr));

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

  RDyMesh *mesh  = user->mesh;
  RDyCell *cells = &mesh->cells;

  PetscInt nbnd_1 = 0, nbnd_2 = 0;
  for (PetscInt icell = 0; icell < mesh->num_cells_local; icell++) {
    PetscReal xc = cells->centroid[icell].X[1];
    PetscReal yc = cells->centroid[icell].X[0];
    if (yc < bu && xc >= bl && xc < br) {
      b_ptr[icell * 3 + 0] = 1.0;
      nbnd_1++;
    } else if (yc >= bd && xc >= bl && xc < br) {
      b_ptr[icell * 3 + 0] = 1.0;
      nbnd_2++;
    }
  }

  PetscCall(VecRestoreArray(user->B, &b_ptr));

  PetscCall(DMGlobalToLocalBegin(user->dm, user->B, INSERT_VALUES, user->localB));
  PetscCall(DMGlobalToLocalEnd(user->dm, user->B, INSERT_VALUES, user->localB));

  PetscCall(PetscPrintf(user->comm, "Building size: bu=%d,bd=%d,bl=%d,br=%d\n", bu, bd, bl, br));
  PetscCall(PetscPrintf(user->comm, "Buildings added sucessfully!\n"));

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

  User user = (User)ptr;

  DM       dm    = user->dm;
  RDyMesh *mesh  = user->mesh;
  RDyCell *cells = &mesh->cells;
  RDyEdge *edges = &mesh->edges;
  // RDyVertex *vertices = &mesh->vertices;

  user->tstep = user->tstep + 1;

  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, user->localX));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, user->localX));
  PetscCall(VecZeroEntries(F));

  // Get pointers to vector data
  PetscScalar *x_ptr, *f_ptr, *b_ptr;
  PetscCall(VecGetArray(user->localX, &x_ptr));
  PetscCall(VecGetArray(F, &f_ptr));
  PetscCall(VecGetArray(user->localB, &b_ptr));

  PetscInt dof = 3;
  for (PetscInt iedge = 0; iedge < mesh->num_edges; iedge++) {
    PetscInt  cellOffset = edges->cell_offset[iedge];
    PetscInt  l          = edges->cell_ids[cellOffset];
    PetscInt  r          = edges->cell_ids[cellOffset + 1];
    PetscReal edgeLen    = edges->length[iedge];

    PetscReal sn, cn;

    PetscBool is_edge_vertical = PETSC_TRUE;
    if (PetscAbs(edges->normal[iedge].V[0]) < 1.e-10) {
      sn = 1.0;
      cn = 0.0;
    } else if (PetscAbs(edges->normal[iedge].V[1]) < 1.e-10) {
      is_edge_vertical = PETSC_FALSE;
      sn               = 0.0;
      cn               = 1.0;
    } else {
      printf("The code only support quad cells with edges that align with x and y axis\n");
      exit(0);
    }

    if (r >= 0 && l >= 0) {
      // Perform computation for an internal edge

      PetscReal hl = x_ptr[l * dof + 0];
      PetscReal hr = x_ptr[r * dof + 0];
      PetscReal bl = b_ptr[l * dof + 0];
      PetscReal br = b_ptr[r * dof + 0];

      if (bl == 0 && br == 0) {
        // Both, left and right cells are not boundary walls
        if (!(hr < user->tiny_h && hl < user->tiny_h)) {
          PetscReal hul   = x_ptr[l * dof + 1];
          PetscReal hvl   = x_ptr[l * dof + 2];
          PetscReal hur   = x_ptr[r * dof + 1];
          PetscReal hvr   = x_ptr[r * dof + 2];
          PetscReal areal = cells->area[l];
          PetscReal arear = cells->area[r];

          PetscReal ur, vr, ul, vl;

          PetscCall(GetVelocityFromMomentum(user->tiny_h, hr, hur, hvr, &ur, &vr));
          PetscCall(GetVelocityFromMomentum(user->tiny_h, hl, hul, hvl, &ul, &vl));

          PetscReal flux[3], amax;
          PetscCall(solver(hl, hr, ul, ur, vl, vr, sn, cn, flux, &amax));

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
        PetscCall(GetVelocityFromMomentum(user->tiny_h, hr, hur, hvr, &ur, &vr));

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

        PetscReal arear = cells->area[r];
        for (PetscInt idof = 0; idof < dof; idof++) {
          if (cells->is_local[r]) f_ptr[r * dof + idof] += flux[idof] * edgeLen / arear;
        }

      } else if (bl == 0 && br == 1) {
        // Left cell is an internal cell and right cell is a boundary wall

        PetscReal hl  = x_ptr[l * dof + 0];
        PetscReal hul = x_ptr[l * dof + 1];
        PetscReal hvl = x_ptr[l * dof + 2];

        PetscReal ul, vl;
        PetscCall(GetVelocityFromMomentum(user->tiny_h, hl, hul, hvl, &ul, &vl));

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

        PetscReal areal = cells->area[l];
        for (PetscInt idof = 0; idof < dof; idof++) {
          if (cells->is_local[l]) f_ptr[l * dof + idof] -= flux[idof] * edgeLen / areal;
        }
      }

    } else if (cells->is_local[l] && b_ptr[l * dof + 0] == 0) {
      // Perform computation for a boundary edge

      PetscBool bnd_cell_order_flipped = PETSC_FALSE;

      if (is_edge_vertical) {
        if (cells->centroid[l].X[1] > edges->centroid[iedge].X[1]) bnd_cell_order_flipped = PETSC_TRUE;
      } else {
        if (cells->centroid[l].X[0] > edges->centroid[iedge].X[0]) bnd_cell_order_flipped = PETSC_TRUE;
      }

      PetscReal hl = x_ptr[l * dof + 0];

      if (!(hl < user->tiny_h)) {
        PetscReal hul = x_ptr[l * dof + 1];
        PetscReal hvl = x_ptr[l * dof + 2];

        PetscReal ul, vl;
        PetscCall(GetVelocityFromMomentum(user->tiny_h, hl, hul, hvl, &ul, &vl));

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

        PetscReal areal = cells->area[l];
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
  PetscCall(VecRestoreArray(user->localX, &x_ptr));
  PetscCall(VecRestoreArray(F, &f_ptr));
  PetscCall(VecRestoreArray(user->localB, &b_ptr));

  if (user->save) {
    char fname[PETSC_MAX_PATH_LEN];
    sprintf(fname, "outputs/ex2_Nx_%d_Ny_%d_dt_%f_%d.dat", user->Nx, user->Ny, user->dt, user->tstep - 1);
    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(user->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(X, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    sprintf(fname, "outputs/ex2_flux_Nx_%d_Ny_%d_dt_%f_%d.dat", user->Nx, user->Ny, user->dt, user->tstep - 1);
    PetscCall(PetscViewerBinaryOpen(user->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(F, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  User user;
  PetscCall(PetscNew(&user));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Create the DM
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "1. CreateDM\n"));
  PetscCall(CreateDM(user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Create vectors for solution and residual
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  Vec X, R;
  PetscCall(DMCreateGlobalVector(user->dm, &X));  // size = dof * number of cells
  PetscCall(VecDuplicate(X, &user->B));
  PetscCall(VecDuplicate(X, &R));
  VecViewFromOptions(X, NULL, "-vec_view");
  PetscCall(DMCreateLocalVector(user->dm, &user->localX));  // size = dof * number of cells
  PetscCall(VecDuplicate(user->localX, &user->localB));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Create the mesh
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2. CreateMesh\n"));
  PetscCall(CreateMesh(user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Initial Condition
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "3. SetInitialCondition\n"));
  PetscCall(SetInitialCondition(user, X));
  {
    char fname[PETSC_MAX_PATH_LEN];
    sprintf(fname, "outputs/ex2_Nx_%d_Ny_%d_dt_%f_IC.dat", user->Nx, user->Ny, user->dt);

    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(user->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(X, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Add buildings
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  if (user->add_building) {
    PetscCall(AddBuildings(user));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Create timestepping solver context
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscReal max_time = user->Nt * user->dt;
  TS        ts;
  PetscCall(TSCreate(user->comm, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSEULER));
  PetscCall(TSSetDM(ts, user->dm));
  PetscCall(TSSetRHSFunction(ts, R, RHSFunction, user));
  PetscCall(TSSetMaxTime(ts, max_time));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(TSSetTimeStep(ts, user->dt));

  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSolve(ts, X));

  if (user->save) {
    char fname[PETSC_MAX_PATH_LEN];
    sprintf(fname, "outputs/ex1_Nx_%d_Ny_%d_dt_%f_%d.dat", user->Nx, user->Ny, user->dt, user->Nt);

    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(user->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(X, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&R));
  PetscCall(DMDestroy(&user->dm));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());

  return 0;
}
