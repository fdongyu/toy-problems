static char help[] = "Dam Break 2D Shallow Water Equation Finite Volume Example.\n";

#include <assert.h>
#include <math.h>
#include <petsc.h>
#include <petscdmda.h>
#include <petscmat.h>
#include <petscts.h>
#include <petscvec.h>

typedef struct _n_User *User;

struct _n_User {
  MPI_Comm  comm;
  DM        da;
  PetscInt  Nt, Nx, Ny;
  PetscReal Lx, Ly, dt, dx, dy;
  PetscReal hu, hd;
  PetscReal max_time, tiny_h;
  PetscInt  dof, rank, size;
  Vec       F, G, B;
  Vec       subdomain;
  PetscInt  xs, ys, xm, ym, xe, ye;
  PetscInt  gxs, gxm, gys, gym, gxe, gye;
  PetscBool debug, add_building;
  PetscInt  save, tstep;
  PetscInt  domain; // domain = 0: create domain from Lx, Ly, dx, dy; domain > 0: predefined domain
};

extern PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode fluxes(PetscScalar ***, PetscScalar ***, PetscScalar ***, User);
extern PetscErrorCode solver(PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscScalar *, PetscScalar *);

static PetscErrorCode SetInitialCondition(Vec X, User user) {
  PetscFunctionBeginUser;
  DM da = user->da;

  PetscBool debug = user->debug;

  // Get pointer to vector data
  PetscScalar ***x_ptr;
  PetscCall(DMDAVecGetArrayDOF(da, X, &x_ptr));

  // Get local grid boundaries
  PetscInt xs, ys, xm, ym;
  PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));

  PetscInt gxs, gys, gxm, gym;
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, 0, &gxm, &gym, 0));
  if (debug) {
    MPI_Comm self;
    self = PETSC_COMM_SELF;
    PetscPrintf(self, "rank = %d, xs = %d, ys = %d, xm = %d, ym = %d\n", user->rank, xs, ys, xm, ym);
    PetscPrintf(self, "rank = %d, gxs = %d, gys = %d, gxm = %d, gym = %d\n", user->rank, gxs, gys, gxm, gym);
  }

  user->xs = xs;
  user->ys = ys;
  user->xm = xm;
  user->ym = ym;

  user->gxs = gxs;
  user->gys = gys;
  user->gxm = gxm;
  user->gym = gym;

  user->xe  = user->xs + user->xm - 1;
  user->ye  = user->ys + user->ym - 1;
  user->gxe = user->gxs + user->gxm - 1;
  user->gye = user->gys + user->gym - 1;

  PetscInt iDam = 95;
  if (user->domain == 1) {
    iDam = 4;
  }
  PetscPrintf(user->comm, "iDam = %d\n", iDam);

  PetscScalar ***b_ptr;
  PetscCall(DMDAVecGetArrayDOF(da, user->B, &b_ptr));

  // Set higher water on the left of the dam
  PetscCall(VecZeroEntries(X));
  for (PetscInt j = ys; j < ys + ym; j = j + 1) {
    for (PetscInt i = xs; i < xs + xm; i = i + 1) {
      if (i < iDam / user->dx) {
        x_ptr[j][i][0] = user->hu;
      } else {
        if (b_ptr[j][i][0] == 1.) {
          x_ptr[j][i][0] = 0.;
        } else {
          x_ptr[j][i][0] = user->hd;
        }
        
      }
    }
  }

  PetscCall(VecZeroEntries(user->F));
  PetscCall(VecZeroEntries(user->G));

  // Restore vectors
  PetscCall(DMDAVecRestoreArrayDOF(da, user->B, &b_ptr));
  PetscCall(DMDAVecRestoreArrayDOF(da, X, &x_ptr));

  PetscPrintf(user->comm, "Initialization sucesses!\n");

  PetscFunctionReturn(0);
}

PetscErrorCode Add_Buildings(User user) {
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(user->comm, "Assiging buildings!\n"));

  DM        da    = user->da;
  PetscBool debug = user->debug;

   // Get local grid boundaries
  PetscInt xs, ys, xm, ym;
  PetscCall(DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0));

  PetscInt gxs, gys, gxm, gym;
  PetscCall(DMDAGetGhostCorners(da, &gxs, &gys, 0, &gxm, &gym, 0));
  if (debug) {
    MPI_Comm self;
    self = PETSC_COMM_SELF;
    PetscPrintf(self, "rank = %d, xs = %d, ys = %d, xm = %d, ym = %d\n", user->rank, xs, ys, xm, ym);
    PetscPrintf(self, "rank = %d, gxs = %d, gys = %d, gxm = %d, gym = %d\n", user->rank, gxs, gys, gxm, gym);
  }

  user->xs = xs;
  user->ys = ys;
  user->xm = xm;
  user->ym = ym;

  user->gxs = gxs;
  user->gys = gys;
  user->gxm = gxm;
  user->gym = gym;
  
  PetscInt bu;
  PetscInt bd;
  PetscInt bl;
  PetscInt br;
  if (user->domain == 0) {
    bu = 30 / user->dx;
    bd = 105 / user->dx;
    bl = 95 / user->dy;
    br = 105 / user->dy;
  } else if (user->domain == 1) {
    bu = 2;
    bd = 4;
    bl = 4;
    br = 6;
  }
  
  PetscScalar ***b_ptr;
  PetscCall(DMDAVecGetArrayDOF(da, user->B, &b_ptr));

  PetscCall(VecZeroEntries(user->B));
  /*

  x represents the reflecting wall,
  | represents the dam that will be broken suddenly.
  hu is the upsatream water dpeth, and hd is the downstream water depth.

  x x x x x x x x x x x
  x         x         x
  x         x         x
  x         |         x
  x         |         x
  x    hu   |    hd   x
  x         x         x
  x         x         x
  x         x         x
  x         x         x
  x x x x x x x x x x x

  */

  PetscCall(PetscPrintf(user->comm, "ys=%d,ym=%d,xs=%d,xm=%d\n", user->ys, user->ym, user->xs, user->xm));

  for (PetscInt j = user->ys; j < user->ys + user->ym; j = j + 1) {
    for (PetscInt i = user->xs; i < user->xs + user->xm; i = i + 1) {
      PetscPrintf(PETSC_COMM_SELF, "i = %d, j = %d\n", i,j);
      if (j < bu && i >= bl && i < br) {
        b_ptr[j][i][0] = 1.;
        PetscPrintf(PETSC_COMM_SELF, "i = %d, j = %d is inactive\n", i,j);
      } else if (j >= bd && i >= bl && i < br) {
        b_ptr[j][i][0] = 1.;
        PetscPrintf(PETSC_COMM_SELF, "i = %d, j = %d is inactive\n", i,j);
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(da, user->B, &b_ptr));

  PetscCall(PetscPrintf(user->comm, "Building size: bu=%d,bd=%d,bl=%d,br=%d\n", bu, bd, bl, br));
  PetscCall(PetscPrintf(user->comm, "Buildings added sucessfully!\n"));

  PetscFunctionReturn(0);
}

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ptr) {
  PetscFunctionBeginUser;

  User user = (User)ptr;

  DM        da   = user->da;
  PetscReal dx   = user->dx;
  PetscReal dy   = user->dy;
  PetscReal area = dx * dy;
  PetscInt  save = user->save;

  user->tstep = user->tstep + 1;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  corrector
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  Vec localX, localF, localG;
  PetscCall(DMGetLocalVector(da, &localX));
  PetscCall(DMGetLocalVector(da, &localF));
  PetscCall(DMGetLocalVector(da, &localG));

  /*
  ! Scatter ghost points to local vector, using the 2-step process
  ! DMGlobalToLocalBegin(), DMGlobalToLocalEnd()
  ! By placing code between these tow statements, computations can be
  ! done while messages are in transition
  */

  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));

  // Get pointers to vector data
  PetscScalar ***x_ptr, ***f_ptr, ***g_ptr, ***f_ptr1;
  PetscCall(DMDAVecGetArrayDOF(da, localX, &x_ptr));
  PetscCall(DMDAVecGetArrayDOF(da, localF, &f_ptr));
  PetscCall(DMDAVecGetArrayDOF(da, localG, &g_ptr));
  PetscCall(DMDAVecGetArrayDOF(da, F, &f_ptr1));

  PetscCall(fluxes(x_ptr, f_ptr, g_ptr, user));

  for (PetscInt j = user->ys; j < user->ys + user->ym; j = j + 1) {
    for (PetscInt i = user->xs; i < user->xs + user->xm; i = i + 1) {
      for (PetscInt k = 0; k < user->dof; k = k + 1) {
        f_ptr1[j][i][k] = -(f_ptr[j][i + 1][k] * dx - f_ptr[j][i][k] * dx + g_ptr[j + 1][i][k] * dy - g_ptr[j][i][k] * dy) / area;
      }
    }
  }

  // Restore vectors
  PetscCall(DMDAVecRestoreArrayDOF(da, localX, &x_ptr));
  PetscCall(DMDAVecRestoreArrayDOF(da, localF, &f_ptr));
  PetscCall(DMDAVecRestoreArrayDOF(da, localG, &g_ptr));
  PetscCall(DMDAVecRestoreArrayDOF(da, F, &f_ptr1));

  PetscCall(DMRestoreLocalVector(da, &localX));
  PetscCall(DMRestoreLocalVector(da, &localF));
  PetscCall(DMRestoreLocalVector(da, &localG));

  if (save == 1) {
    char fname[PETSC_MAX_PATH_LEN];
    sprintf(fname, "outputs/ex1_Nx_%d_Ny_%d_dt_%f_%d.dat", user->Nx, user->Ny, user->dt, user->tstep - 1);

    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(user->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(X, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode fluxes(PetscScalar ***x_ptr, PetscScalar ***f_ptr, PetscScalar ***g_ptr, User user) {
  PetscFunctionBeginUser;

  DM        da    = user->da;
  PetscInt  Nx    = user->Nx;
  PetscInt  Ny    = user->Ny;
  PetscBool debug = user->debug;
  PetscInt  tstep = user->tstep;
  MPI_Comm  self  = PETSC_COMM_SELF;

  Vec localB;
  PetscCall(DMGetLocalVector(da, &localB));
  PetscCall(DMGlobalToLocalBegin(da, user->B, INSERT_VALUES, localB));
  PetscCall(DMGlobalToLocalEnd(da, user->B, INSERT_VALUES, localB));

  PetscScalar ***b_ptr;
  PetscCall(DMDAVecGetArrayDOF(da, localB, &b_ptr));

  PetscReal amax = 0.0;

  for (PetscInt j = user->gys + 1; j < user->gys + user->gym; j = j + 1) {
    for (PetscInt i = user->gxs + 1; i < user->gxs + user->gxm; i = i + 1) {
      /* - - - - - - - - - - - - - - - *
       * Compute fluxes in x-driection *
       * - - - - - - - - - - - - - - - */
      PetscScalar fij[3];

      PetscReal sn = 0.0;
      PetscReal cn = 1.0;
      if (i == 0) {
        // Enforce wall boundary on left side of box (west)
        PetscReal hr = x_ptr[j][i][0];
        if (hr < user->tiny_h) {
          fij[0] = 0.;
          fij[1] = 0.;
          fij[2] = 0.;
        } else {
          PetscReal ur = x_ptr[j][i][1] / hr;
          PetscReal vr = x_ptr[j][i][2] / hr;

          PetscScalar a;
          solver(hr, hr, -ur, ur, vr, vr, sn, cn, fij, &a);
          amax = fmax(a, amax);
        }

      } else if (i == Nx) {
        // Enforce wall boundary on right side of box (west)
        PetscReal hl = x_ptr[j][i - 1][0];
        if (hl < user->tiny_h) {
          fij[0] = 0.;
          fij[1] = 0.;
          fij[2] = 0.;
        } else {
          PetscReal ul = x_ptr[j][i - 1][1] / hl;
          PetscReal vl = x_ptr[j][i - 1][2] / hl;

          PetscScalar a;
          solver(hl, hl, ul, -ul, vl, vl, sn, cn, fij, &a);
          amax = fmax(a, amax);
        }
      } else if (b_ptr[j][i][0] == 1. && b_ptr[j][i - 1][0] == 0.) {
        PetscReal hl = x_ptr[j][i - 1][0];
        if (hl < user->tiny_h) {
          fij[0] = 0.;
          fij[1] = 0.;
          fij[2] = 0.;
        } else {
          PetscReal ul = x_ptr[j][i - 1][1] / hl;
          PetscReal vl = x_ptr[j][i - 1][2] / hl;

          PetscScalar a;
          solver(hl, hl, ul, -ul, vl, vl, sn, cn, fij, &a);
          amax = fmax(a, amax);
        }
      } else if (b_ptr[j][i][0] == 0. && b_ptr[j][i - 1][0] == 1.) {
        PetscReal hr = x_ptr[j][i][0];
        if (hr < user->tiny_h) {
          fij[0] = 0.;
          fij[1] = 0.;
          fij[2] = 0.;
        } else {
          PetscReal ur = x_ptr[j][i][1] / hr;
          PetscReal vr = x_ptr[j][i][2] / hr;

          PetscScalar a;
          solver(hr, hr, -ur, ur, vr, vr, sn, cn, fij, &a);
          amax = fmax(a, amax);
        }
      } else if (b_ptr[j][i][0] == 1. && b_ptr[j][i - 1][0] == 1.) {
        fij[0] = 0.;
        fij[1] = 0.;
        fij[2] = 0.;
      } else {
        PetscReal hl = x_ptr[j][i - 1][0];
        PetscReal hr = x_ptr[j][i][0];
        PetscReal ul, vl, ur, vr;

        if (hl < user->tiny_h) {
          ul = 0.;
          vl = 0.;
        } else {
          ul = x_ptr[j][i - 1][1] / hl;
          vl = x_ptr[j][i - 1][2] / hl;
        }

        if (hr < user->tiny_h) {
          ur = 0.;
          vr = 0.;
        } else {
          ur = x_ptr[j][i][1] / hr;
          vr = x_ptr[j][i][2] / hr;
        }

        if (hl < user->tiny_h && hr < user->tiny_h) {
          fij[0] = 0.;
          fij[1] = 0.;
          fij[2] = 0.;
        } else {
          PetscScalar a;
          solver(hl, hr, ul, ur, vl, vr, sn, cn, fij, &a);
          amax = fmax(a, amax);
        }
      }
      f_ptr[j][i][0] = fij[0];
      f_ptr[j][i][1] = fij[1];
      f_ptr[j][i][2] = fij[2];

      /* - - - - - - - - - - - - - - - *
       * Compute fluxes in y-driection *
       * - - - - - - - - - - - - - - - */
      PetscScalar gij[3];

      sn = 1.0;
      cn = 0.0;
      if (j == 0) {
        PetscReal hr = x_ptr[j][i][0];
        if (hr < user->tiny_h) {
          gij[0] = 0.;
          gij[1] = 0.;
          gij[2] = 0.;
        } else {
          PetscReal ur = x_ptr[j][i][1] / hr;
          PetscReal vr = x_ptr[j][i][2] / hr;

          PetscScalar a;
          solver(hr, hr, ur, ur, -vr, vr, sn, cn, gij, &a);
          amax = fmax(a, amax);
        }

      } else if (j == Ny) {
        PetscReal hl = x_ptr[j - 1][i][0];
        if (hl < user->tiny_h) {
          gij[0] = 0.;
          gij[1] = 0.;
          gij[2] = 0.;
        } else {
          PetscReal ul = x_ptr[j - 1][i][1] / hl;
          PetscReal vl = x_ptr[j - 1][i][2] / hl;

          PetscScalar a;
          solver(hl, hl, ul, ul, vl, -vl, sn, cn, gij, &a);
          amax = fmax(a, amax);
        }
      } else if (b_ptr[j][i][0] == 1. && b_ptr[j - 1][i][0] == 0.) {
        PetscReal hl = x_ptr[j - 1][i][0];
        if (hl < user->tiny_h) {
          gij[0] = 0.;
          gij[1] = 0.;
          gij[2] = 0.;
        } else {
          PetscReal ul = x_ptr[j - 1][i][1] / hl;
          PetscReal vl = x_ptr[j - 1][i][2] / hl;

          PetscScalar a;
          solver(hl, hl, ul, ul, vl, -vl, sn, cn, gij, &a);
          amax = fmax(a, amax);
        }
      } else if (b_ptr[j][i][0] == 0. && b_ptr[j - 1][i][0] == 1.) {
        PetscReal hr = x_ptr[j][i][0];
        if (hr < user->tiny_h) {
          gij[0] = 0.;
          gij[1] = 0.;
          gij[2] = 0.;
        } else {
          PetscReal ur = x_ptr[j][i][1] / hr;
          PetscReal vr = x_ptr[j][i][2] / hr;

          PetscScalar a;
          solver(hr, hr, ur, ur, -vr, vr, sn, cn, gij, &a);
          amax = fmax(a, amax);
        }
      } else if (b_ptr[j][i][0] == 1. && b_ptr[j - 1][i][0] == 1.) {
        gij[0] = 0.;
        gij[1] = 0.;
        gij[2] = 0.;
      } else {
        PetscReal hl = x_ptr[j - 1][i][0];
        PetscReal hr = x_ptr[j][i][0];
        PetscReal ul, vl, ur, vr;

        if (hl < user->tiny_h) {
          ul = 0.;
          vl = 0.;
        } else {
          ul = x_ptr[j - 1][i][1] / hl;
          vl = x_ptr[j - 1][i][2] / hl;
        }

        if (hr < user->tiny_h) {
          ur = 0.;
          vr = 0.;
        } else {
          ur = x_ptr[j][i][1] / hr;
          vr = x_ptr[j][i][2] / hr;
        }

        if (hl < user->tiny_h && hr < user->tiny_h) {
          gij[0] = 0.;
          gij[1] = 0.;
          gij[2] = 0.;
        } else {
          PetscScalar a;
          solver(hl, hr, ul, ur, vl, vr, sn, cn, gij, &a);
          amax = fmax(a, amax);
        }
      }
      g_ptr[j][i][0] = gij[0];
      g_ptr[j][i][1] = gij[1];
      g_ptr[j][i][2] = gij[2];

      if (i == 1 && j == 41 && debug) {
        PetscPrintf(self, "Depth at (%d,%d) is %f, fij=%f, gij=%f.\n", i + 1, j + 1, x_ptr[j][i][0], fij[0], gij[0]);
      }
    }
  }

  PetscPrintf(self, "Time Step = %d, rank = %d, Courant Number = %f\n", tstep, user->rank, amax * user->dt * 2);

  PetscCall(DMDAVecRestoreArrayDOF(da, localB, &b_ptr));
  PetscCall(DMRestoreLocalVector(da, &localB));

  PetscFunctionReturn(0);
}

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

int main(int argc, char **argv) {
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Initialize program and set problem parameters
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  User user;
  PetscCall(PetscNew(&user));
  user->dt       = 0.04;
  user->max_time = 7.2;
  user->dof      = 3;  // h, uh, vh
  user->comm     = PETSC_COMM_WORLD;
  user->debug    = PETSC_FALSE;
  user->save     = 0;  // save = 1: save outputs for each time step; save = 2: save outputs at last time step

  MPI_Comm_size(user->comm, &user->size);
  MPI_Comm_rank(user->comm, &user->rank);
  PetscPrintf(user->comm, "Running on %d processors! \n", user->size);
  /*Default number of cells (box), cellsize in x-direction, and y-direction */
  user->Lx     = 200;
  user->Ly     = 200;
  user->dx     = 1.0;
  user->dy     = 1.0;
  user->hu     = 10.0;  // water depth for the upstream of dam   [m]
  user->hd     = 5.0;   // water depth for the downstream of dam [m]
  user->tiny_h = 1e-7;
  user->domain = 0;     

  PetscOptionsBegin(user->comm, NULL, "2D Mesh Options", "");
  {
    PetscCall(PetscOptionsReal("-t", "simulation time", "", user->max_time, &user->max_time, NULL));
    PetscCall(PetscOptionsReal("-Lx", "Length in X", "", user->Lx, &user->Lx, NULL));
    PetscCall(PetscOptionsReal("-Ly", "Length in Y", "", user->Ly, &user->Ly, NULL));
    PetscCall(PetscOptionsReal("-dx", "dx", "", user->dx, &user->dx, NULL));
    PetscCall(PetscOptionsReal("-dy", "dy", "", user->dy, &user->dy, NULL));
    PetscCall(PetscOptionsReal("-hu", "hu", "", user->hu, &user->hu, NULL));
    PetscCall(PetscOptionsReal("-hd", "hd", "", user->hd, &user->hd, NULL));
    PetscCall(PetscOptionsReal("-dt", "dt", "", user->dt, &user->dt, NULL));
    PetscCall(PetscOptionsBool("-b", "Add buildings", "", user->add_building, &user->add_building, NULL));
    PetscCall(PetscOptionsBool("-debug", "debug", "", user->debug, &user->debug, NULL));
    PetscCall(PetscOptionsInt("-save", "save outputs", "", user->save, &user->save, NULL));
    PetscCall(PetscOptionsInt("-domain", "domain options", "", user->domain, &user->domain, NULL));
  }
  PetscOptionsEnd();
  assert(user->hu >= 0.);
  assert(user->hd >= 0.);

  PetscPrintf(user->comm, "Using domain option = %d; \n", user->domain);

  if (user->domain == 0) {
    user->Nx = user->Lx / user->dx;
    user->Ny = user->Ly / user->dy;
  } else if (user->domain == 1) {
    user->Lx = 10; // [m]
    user->Ly = 5;  // [m]
    user->dx = 1;  // [m]
    user->dy = 1;  // [m]
    user->Nx = user->Lx / user->dx;
    user->Ny = user->Ly / user->dy;
  }
  else {
    exit(0);
  }
  
  user->Nt = user->max_time / user->dt;
  PetscPrintf(user->comm, "Max simulation time is %f; \n", user->max_time);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Initialize DMDA
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscCall(DMDACreate2d(user->comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_BOX, user->Nx, user->Ny, PETSC_DECIDE, PETSC_DECIDE,
                         user->dof, 1, NULL, NULL, &user->da));
  PetscCall(DMSetFromOptions(user->da));
  PetscCall(DMSetUp(user->da));
  PetscCall(DMDASetUniformCoordinates(user->da, 0.0, user->Lx, 0.0, user->Ly, 0.0, 0.0));

  /*
  {
      PetscInt dimEmbed, i;
      PetscInt nCoords;
      PetscScalar *coords;
      Vec coordinates;

      PetscCall(DMGetCoordinatesLocal(user->da,&coordinates));
      PetscCall(DMGetCoordinateDim(user->da,&dimEmbed));
      PetscCall(VecGetLocalSize(coordinates,&nCoords));
      PetscCall(VecGetArray(coordinates,&coords));

      PetscPrintf(PETSC_COMM_SELF,"nCoords = %d, dimEmbed = %d\n",nCoords,dimEmbed);
      for (i = 0; i < nCoords; i += dimEmbed) {
        PetscInt j;
        PetscScalar *coord = &coords[i];
        PetscPrintf(PETSC_COMM_SELF,"i = %d: x = %f, y = %f\n",i,coord[0],coord[1]);
      }
      PetscCall(VecRestoreArray(coordinates,&coords));

      PetscInt numCells, numCellsX;
      DMDAGetNumCells(user->da, &numCellsX, NULL, NULL, &numCells);
      PetscPrintf(PETSC_COMM_SELF,"numCells = %d, numCellsX = %d\n",numCells,numCellsX);
  }
  */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Extract global vectors from DMDA
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  Vec X, R;
  PetscCall(DMCreateGlobalVector(user->da, &X));  // size = dof * number of cells
  PetscCall(VecDuplicate(X, &user->F));
  PetscCall(VecDuplicate(X, &user->G));
  PetscCall(VecDuplicate(X, &user->B));
  PetscCall(VecDuplicate(X, &R));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Add buildings
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  if (user->add_building) {
    PetscCall(Add_Buildings(user));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Initial Condition
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  SetInitialCondition(X, user);
  {
    char fname[PETSC_MAX_PATH_LEN];
    sprintf(fname, "outputs/ex1_Nx_%d_Ny_%d_dt_%f_IC.dat", user->Nx, user->Ny, user->dt);

    if (user->save > 0) {
      PetscViewer viewer;
      PetscCall(PetscViewerBinaryOpen(user->comm, fname, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(X, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Set sub domains
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  // VecCreateMPI(user->comm, PetscInt n, PETSC_DETERMINE, user->subdomain);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Create timestepping solver context
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  TS ts;
  PetscCall(TSCreate(user->comm, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSEULER));
  PetscCall(TSSetDM(ts, user->da));
  PetscCall(TSSetRHSFunction(ts, R, RHSFunction, user));
  PetscCall(TSSetMaxTime(ts, user->max_time));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(TSSetTimeStep(ts, user->dt));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Sets various TS parameters from user options
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Solver nonlinear system
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  PetscCall(TSSolve(ts, X));

  if (user->save == 2) {
    char fname[PETSC_MAX_PATH_LEN];
    sprintf(fname, "outputs/ex1_Nx_%d_Ny_%d_dt_%f_%d.dat", user->Nx, user->Ny, user->dt, user->tstep);

    PetscViewer viewer;
    PetscCall(PetscViewerBinaryOpen(user->comm, fname, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(X, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&R));
  PetscCall(VecDestroy(&user->F));
  PetscCall(VecDestroy(&user->G));
  PetscCall(VecDestroy(&user->B));
  PetscCall(PetscFinalize());

  return 0;
}
