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
  PetscInt *num_edges;          /* number of edges sharing the vertex                   */

  PetscInt *edge_offset;          /* offset for edge IDs that share the vertex                       */
  PetscInt *cell_offset; /* offset for internal cell IDs that share the vertex              */

  PetscInt *edge_ids;          /* edge IDs that share the vertex                       */
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
  PetscReal dt, hx, hy;
  PetscReal Lx, Ly;
  PetscReal hu, hd;
  PetscReal tiny_h;
  PetscInt  dof, rank, comm_size;
  Vec       B;
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
  user->Lx     = user->Nx * 1.0;
  user->Ly     = user->Ny * 1.0;
  user->hx     = 1.0;
  user->hy     = 1.0;
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
    PetscCall(PetscOptionsReal("-hx", "dx", "", user->hx, &user->hx, NULL));
    PetscCall(PetscOptionsReal("-hy", "dy", "", user->hy, &user->hy, NULL));
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
static PetscErrorCode PopulateCellsFromDM(DM dm, RDyCell *cells) {

  PetscFunctionBegin;

  PetscInt cStart, cEnd;
  PetscInt eStart, eEnd;
  DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);

  for (PetscInt c = cStart; c < cEnd; c++) {

    PetscInt icell = c - cStart;
    PetscInt  gref, junkInt;
    PetscInt dim=2;
    PetscReal centroid[dim], normal[dim];
    PetscCall(DMPlexGetPointGlobal(dm, c, &gref, &junkInt));
    DMPlexComputeCellGeometryFVM(dm, c, &cells->area[icell], &centroid[0], &normal[0]);

    for (PetscInt idim=0; idim<dim; idim++) {
      cells->centroid[icell].X[idim] = centroid[idim];
    }

    PetscInt  pSize;
    PetscInt *p        = NULL;
    PetscInt  use_cone = PETSC_TRUE;

    cells->num_vertices[icell] = 0;
    cells->num_edges[icell] = 0;
    if (gref >= 0) {
      cells->is_local[icell] = PETSC_TRUE;
    } else {
      cells->is_local[icell] = PETSC_FALSE;
    }

    PetscCall(DMPlexGetTransitiveClosure(dm, c, use_cone, &pSize, &p));
    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], eStart, eEnd)) {
        PetscInt offset = cells->edge_offset[icell];
        PetscInt index = offset + cells->num_edges[icell];
        cells->edge_ids[index] = p[i];
        cells->num_edges[icell]++;
      } else {
        PetscInt offset = cells->vertex_offset[icell];
        PetscInt index = offset + cells->num_vertices[icell];
        cells->vertex_ids[index] = p[i];
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

  PetscInt eStart, eEnd;
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);

  for (PetscInt e = eStart; e < eEnd; e++) {

    PetscInt iedge = e - eStart;
    PetscInt dim = 2;
    PetscReal centroid[dim], normal[dim];
    DMPlexComputeCellGeometryFVM(dm, e, &edges->length[iedge], &centroid[0], &normal[0]);

    for (PetscInt idim=0; idim<dim; idim++) {
      edges->centroid[iedge].X[idim] = centroid[idim];
      edges->normal[iedge].V[idim] = normal[idim];
    }

    // edge-to-vertex
    PetscInt  pSize;
    PetscInt *p        = NULL;
    PetscInt  use_cone = PETSC_TRUE;
    PetscCall(DMPlexGetTransitiveClosure(dm, e, use_cone, &pSize, &p));
    assert(pSize == 3);
    PetscInt index = iedge*2;
    edges->vertex_ids[index+0] = p[2];
    edges->vertex_ids[index+1] = p[4];
    PetscCall(DMPlexRestoreTransitiveClosure(dm, e, use_cone, &pSize, &p));

    // edge-to-cell
    edges->num_cells[iedge] = 0;
    PetscCall(DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &pSize, &p));
    assert(pSize == 2 || pSize == 3);
    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      PetscInt offset = edges->cell_offset[iedge];
      PetscInt index = offset + edges->num_cells[iedge];
      edges->cell_ids[index] = p[i];
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

  PetscInt eStart, eEnd;
  PetscInt vStart, vEnd;
  DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);
  DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);

  PetscSection coordSection;
  Vec          coordinates;
  DMGetCoordinateSection(dm, &coordSection);
  DMGetCoordinatesLocal(dm, &coordinates);
  PetscReal *coords;
  VecGetArray(coordinates, &coords);

  for (PetscInt v = vStart; v < vEnd; v++) {

    PetscInt ivertex = v - vStart;
    PetscInt  pSize;
    PetscInt *p = NULL;

    PetscCall(DMPlexGetTransitiveClosure(dm, v, PETSC_FALSE, &pSize, &p));

    PetscInt coordOffset, dim = 2;
    PetscSectionGetOffset(coordSection, v, &coordOffset);
    for (PetscInt idim=0; idim<dim; idim++) {
      vertices->coordinate[ivertex].X[idim] = coords[coordOffset+idim];
    }

    vertices->num_edges[ivertex] = 0;
    vertices->num_cells[ivertex] = 0;

    for (PetscInt i = 2; i < pSize * 2; i += 2) {
      if (IsClosureWithinBounds(p[i], eStart, eEnd)) {
        PetscInt offset = vertices->edge_offset[ivertex];
        PetscInt index = offset + vertices->num_edges[ivertex];
        vertices->edge_ids[index] = p[i];
        vertices->num_edges[ivertex]++;
      } else {
        PetscInt offset = vertices->cell_offset[ivertex];
        PetscInt index = offset + vertices->num_cells[ivertex];
        vertices->cell_ids[index] = p[i];
        vertices->num_cells[ivertex]++;
      }
    }

    PetscCall(DMPlexRestoreTransitiveClosure(dm, v, PETSC_FALSE, &pSize, &p));
  }

  VecRestoreArray(coordinates, &coords);

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
  PetscCall(PopulateCellsFromDM(user->dm, &mesh_ptr->cells));
  PetscCall(PopulateEdgesFromDM(user->dm, &mesh_ptr->edges));
  PetscCall(PopulateVerticesFromDM(user->dm, &mesh_ptr->vertices));

  return 0;
}

int main(int argc, char **argv) {

  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  User user;
  PetscCall(PetscNew(&user));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, user));

  PetscCall(CreateDM(user));

  Vec X, R;
  PetscCall(DMCreateGlobalVector(user->dm, &X));  // size = dof * number of cells
  PetscCall(VecDuplicate(X, &user->B));
  VecViewFromOptions(X, NULL, "-vec_view");

  PetscCall(CreateMesh(user));

  PetscCall(PetscFinalize());

  return 0;
}
