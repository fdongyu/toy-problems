static char help[] = "Dam Break 2D Shallow Water Equation Finite Volume Example.\n";

#include <petscts.h>
#include <petsc.h>
#include <petscdmda.h>
#include <petscvec.h>
#include <petscmat.h>
#include <math.h>

typedef struct _n_User *User;

struct _n_User {
  	MPI_Comm  comm;
	DM        da;
	PetscInt  Nt, Nx, Ny;
	PetscReal dt, hx, hy;
    PetscReal hu, hd;
	PetscInt  dof, rank, size;
	Vec       F, G, B;
    Vec       subdomain;
	PetscInt  xs, ys, xm, ym, xe, ye;
	PetscInt  gxs, gxm, gys, gym, gxe, gye;
    PetscBool debug, save;
    PetscInt  tstep;
};

extern PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode fluxes(PetscScalar ***, PetscScalar ***, PetscScalar ***, User);
extern PetscErrorCode solver(PetscReal, PetscReal, PetscReal, PetscReal,  \
                             PetscReal, PetscReal, PetscReal, PetscReal,  \
                             PetscScalar *, PetscScalar *);

static PetscErrorCode SetInitialCondition(Vec X, User user)
{
	DM             da;
    PetscInt       i, j, xs, ys, xm, ym, Nx, Ny;
    PetscInt       gxs, gys, gxm, gym;
    PetscScalar ***x_ptr;
    PetscBool      debug;

	PetscFunctionBeginUser;
	da    = user->da;
    Nx    = user->Nx;
    Ny    = user->Ny;
    debug = user->debug;
    // Get pointer to vector data 
    DMDAVecGetArrayDOF(da, X, &x_ptr);

    // Get local grid boundaries 
    DMDAGetCorners(da, &xs, &ys, 0, &xm, &ym, 0);
    DMDAGetGhostCorners(da,&gxs,&gys,0,&gxm,&gym,0);
    if (debug) 
    {	
    	MPI_Comm self;
    	self = PETSC_COMM_SELF;
    	PetscPrintf(self,"rank = %d, xs = %d, ys = %d, xm = %d, ym = %d\n",    \
    	                  		 user->rank, xs, ys, xm, ym);
    	PetscPrintf(self,"rank = %d, gxs = %d, gys = %d, gxm = %d, gym = %d\n",\
    	                  	 	 user->rank, gxs, gys, gxm, gym);	
    }

    user->xs = xs;
    user->ys = ys;
    user->xm = xm;
    user->ym = ym;

    user->gxs = gxs;
    user->gys = gys;
    user->gxm = gxm;
    user->gym = gym;

    user->xe  = user->xs  + user->xm  - 1;
    user->ye  = user->ys  + user->ym  - 1;
    user->gxe = user->gxs + user->gxm - 1;
    user->gye = user->gys + user->gym - 1;

    // Set higher water on the left of the dam
    VecZeroEntries(X);
	for (j = ys; j < ys + ym; j = j + 1) {
		for (i = xs; i < xs + xm; i = i + 1) {
			if (j < 95) {
				x_ptr[j][i][0] = user->hu;
			}
			else {
				x_ptr[j][i][0] = user->hd;
			}
		}
	}

    VecZeroEntries(user->F);
    VecZeroEntries(user->G);

    // Restore vectors 
    DMDAVecRestoreArrayDOF(da, X, &x_ptr);

    PetscPrintf(user->comm,"Initialization sucesses!\n");

	PetscFunctionReturn(0);
}

PetscErrorCode Add_Buildings(User user)
{
    // Local variables
    DM              da;
    PetscInt        i, j, Nx, Ny, xmin, xmax, ymin, ymax;
    PetscScalar  ***b_ptr;

    PetscFunctionBeginUser;
    da = user->da;
    Nx = user->Nx;
    Ny = user->Ny;

    xmin = Nx/2 - 3;
    xmax = Nx/2 + 3;
    ymin = Ny/2 - 3;
    ymax = Ny/2 + 3;

    DMDAVecGetArrayDOF  (da, user->B, &b_ptr);
    VecZeroEntries(user->B);
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
    for (j = user->ys; j < user->ys + user->ym; j = j + 1) {
        for (i = user->xs; i < user->xs + user->xm; i = i + 1) {
            if (i < 30 && j >= 95 && j < 105) {
                b_ptr[j][i][0] = 1.;
            }
            else if (i >= 105 && j >= 95 && j < 105) {
                b_ptr[j][i][0] = 1.;
            }
        }
    }

    DMDAVecRestoreArrayDOF(da, user->B, &b_ptr);

    PetscPrintf(user->comm,"Building size: xmin=%d,xmax=%d,ymin=%d,ymax=%d\n", \
                            xmin,   xmax,   ymin,   ymax);
    PetscPrintf(user->comm,"Buildings added sucessfully!\n");

    PetscFunctionReturn(0);
}

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ptr)
{
	// Local variables
	User           user = (User)ptr;
	DM             da;
	PetscInt       i, j, k, Nx, Ny;
	Vec            localX, localF, localG;
	PetscScalar ***x_ptr, ***f_ptr, ***g_ptr, ***f_ptr1;
    PetscBool      save;
    PetscViewer    viewer;


	PetscFunctionBeginUser;
	da   = user->da;
    Nx   = user->Nx;
    Ny   = user->Ny;
    save = user->save;

    user->tstep = user->tstep + 1;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
  	 *  corrector
  	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
    DMGetLocalVector(da, &localX);
    DMGetLocalVector(da, &localF);
    DMGetLocalVector(da, &localG);

    /*
    ! Scatter ghost points to local vector, using the 2-step process
    ! DMGlobalToLocalBegin(), DMGlobalToLocalEnd()
    ! By placing code between these tow statements, computations can be
    ! done while messages are in transition
    */

    DMGlobalToLocalBegin(da, X,       INSERT_VALUES, localX);
    DMGlobalToLocalEnd  (da, X,       INSERT_VALUES, localX);
    DMGlobalToLocalBegin(da, user->F, INSERT_VALUES, localF);
    DMGlobalToLocalEnd  (da, user->F, INSERT_VALUES, localF);
    DMGlobalToLocalBegin(da, user->G, INSERT_VALUES, localG);
    DMGlobalToLocalEnd  (da, user->G, INSERT_VALUES, localG);

    // Get pointers to vector data
    DMDAVecGetArrayDOF(da,localX, &x_ptr );
    DMDAVecGetArrayDOF(da,localF, &f_ptr );
    DMDAVecGetArrayDOF(da,localG, &g_ptr );
    DMDAVecGetArrayDOF(da,F     , &f_ptr1);

    fluxes(x_ptr, f_ptr, g_ptr, user);

    for (j = user->ys; j < user->ys + user->ym; j = j + 1) {
    	for (i = user->xs; i < user->xs + user->xm; i = i + 1) {
    		for (k = 0; k < user->dof; k = k + 1) {
    			f_ptr1[j][i][k] = -(f_ptr[j][i+1][k] - f_ptr[j][i][k]          \
                                  + g_ptr[j+1][i][k] - g_ptr[j][i][k]);
    		}
    	}
    }

    // Restore vectors 
    DMDAVecRestoreArrayDOF(da, localX, &x_ptr );
    DMDAVecRestoreArrayDOF(da, localF, &f_ptr );
    DMDAVecRestoreArrayDOF(da, localG, &g_ptr );
    DMDAVecRestoreArrayDOF(da, F     , &f_ptr1);

    DMRestoreLocalVector(da, &localX);
    DMRestoreLocalVector(da, &localF);
    DMRestoreLocalVector(da, &localG);

    if (save) {
        char fname[PETSC_MAX_PATH_LEN];
        sprintf(fname, "outputs/ex1_%d.dat", user->tstep);
        PetscViewerBinaryOpen(user->comm, fname, FILE_MODE_WRITE, &viewer);
        VecView(X,viewer);
        PetscViewerDestroy(&viewer);
    }
	PetscFunctionReturn(0);
}

PetscErrorCode fluxes(PetscScalar ***x_ptr, PetscScalar ***f_ptr, PetscScalar ***g_ptr, User user)
{
    // Local variables
    MPI_Comm        self;
    DM              da;
    Vec             localB;
    PetscReal       sn, cn, amax, hl, hr, ul, ur, vl, vr;
    PetscInt        i, j, Nx, Ny;
    PetscScalar  ***b_ptr;
    PetscScalar     fij[3], gij[3];
    PetscScalar     a;
    PetscBool       debug;
    PetscInt        tstep;

    PetscFunctionBeginUser;
    da    = user->da;
    Nx    = user->Nx;
    Ny    = user->Ny;
    debug = user->debug;
    tstep = user->tstep;
    self  = PETSC_COMM_SELF;

    DMGetLocalVector    (da, &localB);
    DMGlobalToLocalBegin(da, user->B, INSERT_VALUES, localB);
    DMGlobalToLocalEnd  (da, user->B, INSERT_VALUES, localB);
    DMDAVecGetArrayDOF  (da,localB, &b_ptr);

    amax = 0.0;

    for (j = user->gys+1; j < user->gys + user->gym; j = j + 1) {
        for (i = user->gxs+1; i < user->gxs + user->gxm; i = i + 1) {

            /* - - - - - - - - - - - - - - - *
             * Compute fluxes in x-driection *
             * - - - - - - - - - - - - - - - */

            sn = 0.0;
            cn = 1.0;
            if (i == 0) { 
            // Enforce wall boundary on left side of box (west)
                hr = x_ptr[j][i][0];
                ur = x_ptr[j][i][1]/hr;
                vr = x_ptr[j][i][2]/hr;
                solver(hr, hr, -ur, ur, vr, vr, sn, cn, fij, &a);
                amax = fmax(a,amax);
            }
            else if (i == Nx) { 
            // Enforce wall boundary on right side of box (west)
                hl = x_ptr[j][i-1][0];
                ul = x_ptr[j][i-1][1]/hl;
                vl = x_ptr[j][i-1][2]/hl;
                solver(hl, hl, ul, -ul, vl, vl, sn, cn, fij, &a);
                amax = fmax(a,amax);
            }
            else if (b_ptr[j][i][0] == 1. && b_ptr[j][i-1][0] == 0.) {
                hl = x_ptr[j][i-1][0];
                ul = x_ptr[j][i-1][1]/hl;
                vl = x_ptr[j][i-1][2]/hl;
                solver(hl, hl, ul, -ul, vl, vl, sn, cn, fij, &a);
                amax = fmax(a,amax);
            }
            else if (b_ptr[j][i][0] == 0. && b_ptr[j][i-1][0] == 1.) {
                hr = x_ptr[j][i][0];
                ur = x_ptr[j][i][1]/hr;
                vr = x_ptr[j][i][2]/hr;
                solver(hr, hr, -ur, ur, vr, vr, sn, cn, fij, &a);
                amax = fmax(a,amax);
            }
            else if (b_ptr[j][i][0] == 1. && b_ptr[j][i-1][0] == 1.) {
                fij[0] = 0.;
                fij[1] = 0.;
                fij[2] = 0.;
            }
            else {
                hl = x_ptr[j][i-1][0];
                hr = x_ptr[j][i][0];
                ul = x_ptr[j][i-1][1]/hl;
                ur = x_ptr[j][i][1]/hr;
                vl = x_ptr[j][i-1][2]/hl;
                vr = x_ptr[j][i][2]/hr;
                solver(hl, hr, ul, ur, vl, vr, sn, cn, fij, &a);
                amax = fmax(a,amax);
            }
            f_ptr[j][i][0] = fij[0];
            f_ptr[j][i][1] = fij[1];
            f_ptr[j][i][2] = fij[2];

            /* - - - - - - - - - - - - - - - *
             * Compute fluxes in y-driection *
             * - - - - - - - - - - - - - - - */

            sn = 1.0;
            cn = 0.0;
            if (j == 0) {
                hr = x_ptr[j][i][0];
                ur = x_ptr[j][i][1]/hr;
                vr = x_ptr[j][i][2]/hr;
                solver(hr, hr, ur, ur, -vr, vr, sn, cn, gij, &a);
                amax = fmax(a, amax);
            }
            else if (j == Ny) {
                hl = x_ptr[j-1][i][0];
                ul = x_ptr[j-1][i][1]/hl;
                vl = x_ptr[j-1][i][2]/hl;
                solver(hl, hl, ul, ul, vl, -vl, sn, cn, gij, &a);
                amax = fmax(a, amax);
            }
            else if (b_ptr[j][i][0] == 1. && b_ptr[j-1][i][0] == 0.) {
                hl = x_ptr[j-1][i][0];
                ul = x_ptr[j-1][i][1]/hl;
                vl = x_ptr[j-1][i][2]/hl;
                solver(hl, hl, ul, ul, vl, -vl, sn, cn, gij, &a);
                amax = fmax(a, amax);
            }
            else if (b_ptr[j][i][0] == 0. && b_ptr[j-1][i][0] == 1.) {
                hr = x_ptr[j][i][0];
                ur = x_ptr[j][i][1]/hr;
                vr = x_ptr[j][i][2]/hr;
                solver(hr, hr, ur, ur, -vr, vr, sn, cn, gij, &a);
                amax = fmax(a, amax);
            }
            else if (b_ptr[j][i][0] == 1. && b_ptr[j-1][i][0] == 1.) {
                gij[0] = 0.;
                gij[1] = 0.;
                gij[2] = 0.;
            }
            else {
                hl = x_ptr[j-1][i][0];
                hr = x_ptr[j][i][0];
                ul = x_ptr[j-1][i][1]/hl;
                ur = x_ptr[j][i][1]/hr;
                vl = x_ptr[j-1][i][2]/hl;
                vr = x_ptr[j][i][2]/hr;
                solver(hl, hr, ul, ur, vl, vr, sn, cn, gij, &a);
                amax = fmax(a,amax);
            }
            g_ptr[j][i][0] = gij[0];
            g_ptr[j][i][1] = gij[1];
            g_ptr[j][i][2] = gij[2];

            if (i == 1 && j == 41 && debug) {
                PetscPrintf(self,"Depth at (%d,%d) is %f, fij=%f, gij=%f.\n",  \
                            i+1,j+1,x_ptr[j][i][0],fij[0],gij[0]);
            }

        }
    }

    PetscPrintf(self,"Time Step = %d, rank = %d, Courant Number = %f\n", \
                            tstep, user->rank, amax*user->dt*2);

    DMDAVecRestoreArrayDOF(da, localB, &b_ptr);
    DMRestoreLocalVector  (da, &localB);

    PetscFunctionReturn(0);
}

PetscErrorCode solver(PetscReal hl, PetscReal hr, PetscReal ul, PetscReal ur,  \
                      PetscReal vl, PetscReal vr, PetscReal sn, PetscReal cn,  \
                      PetscScalar *fij, PetscScalar *amax)
{
    // Local variables
    PetscReal duml, dumr, cl, cr, hhat, uhat, vhat, chat, uperp, dh, du, dv;
    PetscReal duperp, dW[3], al1, al3, ar1, ar3, R[3][3], da1, da3, a1, a2, a3;
    PetscReal dupar, uperpl, uperpr, A[3][3], FL[3], FR[3];
    PetscReal grav;
    PetscInt  i, j;

    grav = 9.806;

    PetscFunctionBeginUser;

    // Compute Roe averages
    duml   = pow(hl,0.5);
    dumr   = pow(hr,0.5);
    cl     = pow(grav*hl,0.5);
    cr     = pow(grav*hr,0.5);
    hhat   = duml*dumr;
    uhat   = (duml*ul + dumr*ur)/(duml + dumr);
    vhat   = (duml*vl + dumr*vr)/(duml + dumr);
    chat   = pow(0.5*grav*(hl + hr),0.5);
    uperp  = uhat*cn + vhat*sn;
    dh     = hr - hl;
    du     = ur - ul;
    dv     = vr - vl;
    dupar  = -du*sn + dv*cn;
    duperp = du*cn + dv*sn;
    dW[0]  = 0.5*(dh - hhat*duperp/chat);
    dW[1]  = hhat*dupar;
    dW[2]  = 0.5*(dh + hhat*duperp/chat);

    uperpl = ul*cn + vl*sn;
    uperpr = ur*cn + vr*sn;
    al1 = uperpl - cl;
    al3 = uperpl + cl;
    ar1 = uperpr - cr;
    ar3 = uperpr + cr;

    R[0][0] = 1.0;            R[0][1] = 0.0; R[0][2] = 1.0; 
    R[1][0] = uhat - chat*cn; R[1][1] = -sn; R[1][2] = uhat + chat*cn;
    R[2][0] = vhat - chat*sn; R[2][1] =  cn; R[2][2] = vhat + chat*sn;

    da1 = fmax(0.0, 2.0*(ar1 - al1));
    da3 = fmax(0.0, 2.0*(ar3 - al3));
    a1  = fabs(uperp - chat);
    a2  = fabs(uperp);
    a3  = fabs(uperp + chat);

    // Critical flow fix
    if ( a1 < da1 ) {
        a1 = 0.5*(a1*a1/da1 + da1);
    }
    if ( a3 < da3 ) {
        a3 = 0.5*(a3*a3/da3 + da3);
    }

    // Compute interface flux
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            A[i][j] = 0.0;
        }
    }
    A[0][0] = a1;
    A[1][1] = a2;
    A[2][2] = a3;

    FL[0] = uperpl*hl;
    FL[1] = ul*uperpl*hl + 0.5*grav*hl*hl*cn;
    FL[2] = vl*uperpl*hl + 0.5*grav*hl*hl*sn;

    FR[0] = uperpr*hr;
    FR[1] = ur*uperpr*hr + 0.5*grav*hr*hr*cn;
    FR[2] = vr*uperpr*hr + 0.5*grav*hr*hr*sn;

    //fij = 0.5*(FL + FR - matmul(R,matmul(A,dW))
    fij[0] = 0.5* (FL[0] + FR[0] - R[0][0]*A[0][0]*dW[0] - R[0][1]*A[1][1]*dW[1] - R[0][2]*A[2][2]*dW[2]);
    fij[1] = 0.5* (FL[1] + FR[1] - R[1][0]*A[0][0]*dW[0] - R[1][1]*A[1][1]*dW[1] - R[1][2]*A[2][2]*dW[2]);
    fij[2] = 0.5* (FL[2] + FR[2] - R[2][0]*A[0][0]*dW[0] - R[2][1]*A[1][1]*dW[1] - R[2][2]*A[2][2]*dW[2]);

    *amax = chat + fabs(uperp);

    PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
	TS                ts;
  	Vec               X, R;
  	//Mat               h; // water depth [m]
 	DM                da;
  	User              user;
  	PetscErrorCode    ierr;
  	PetscReal         max_time;
  	PetscBool         add_building;
  	PetscViewer       viewer;
  	//char              outputfile[80];

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
   *  Initialize program and set problem parameters
   * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
	PetscCall(PetscInitialize(&argc, &argv, (char*) 0, help));

	PetscCall(PetscNew(&user));
  	user->dt    = 0.04;
  	user->Nt    = 180;
  	user->dof   = 3;  // h, uh, vh
  	user->comm  = PETSC_COMM_WORLD;
    user->debug = PETSC_FALSE;
    user->debug = PETSC_FALSE;

  	MPI_Comm_size(user->comm,&user->size);
    MPI_Comm_rank(user->comm,&user->rank);
    PetscPrintf(user->comm,"Running on %d processors! \n",user->size);
    /*Default number of cells (box), cellsize in x-direction, and y-direction */
    user->Nx = 200;
    user->Ny = 200;
  	user->hx = 1.0;
  	user->hy = 1.0;
    user->hu = 10.0; // water depth for the upstream of dam   [m]
    user->hd = 5.0;  // water depth for the downstream of dam [m]

  	ierr = PetscOptionsBegin(user->comm,NULL,"2D Mesh Options","");
  	PetscCall(ierr);
   	{
   		PetscCall(PetscOptionsInt("-Nx","Number of cells in X","",user->Nx,&user->Nx,NULL));
   		PetscCall(PetscOptionsInt("-Ny","Number of cells in Y","",user->Ny,&user->Ny,NULL));
   		PetscCall(PetscOptionsInt("-Nt","Number of time steps","",user->Nt,&user->Nt,NULL));
	    PetscCall(PetscOptionsReal("-hx","dx","",user->hx,&user->hx,NULL));
	    PetscCall(PetscOptionsReal("-hy","dy","",user->hy,&user->hy,NULL));
        PetscCall(PetscOptionsReal("-hu","hu","",user->hu,&user->hu,NULL));
        PetscCall(PetscOptionsReal("-hd","hd","",user->hd,&user->hd,NULL));
	    PetscCall(PetscOptionsReal("-dt","dt","",user->dt,&user->dt,NULL));
	    PetscCall(PetscOptionsBool("-b","Add buildings","",add_building,&add_building,NULL));
        PetscCall(PetscOptionsBool("-debug","debug","",user->debug,&user->debug,NULL));
        PetscCall(PetscOptionsBool("-save","save outputs","",user->save,&user->save,NULL));
  	}
  	ierr = PetscOptionsEnd();
  	PetscCall(ierr);
  	max_time = user->Nt * user->dt;
  	PetscPrintf(user->comm,"Max simulation time is %f\n",max_time);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
  	 *  Initialize DMDA
  	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
  	DMDACreate2d(user->comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,         \
       DMDA_STENCIL_BOX, user->Nx, user->Ny, PETSC_DECIDE, PETSC_DECIDE,       \
       user->dof, 1, NULL, NULL, &user->da);
  	DMSetFromOptions(user->da);
  	DMSetUp(user->da);
    DMDASetUniformCoordinates(user->da, 0.0, user->Nx*user->hx,                \
                                        0.0, user->Ny*user->hy, 0.0, 0.0);

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
  	DMCreateGlobalVector(user->da, &X); // size = dof * number of cells
	VecDuplicate(X, &user->F);
	VecDuplicate(X, &user->G);
    VecDuplicate(X, &user->B);
    VecDuplicate(X, &R);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
  	 *  Initial Condition
  	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
    SetInitialCondition(X, user);
    {
    	char fname[PETSC_MAX_PATH_LEN] = "outputs/ex1_IC.dat";
	    PetscViewerBinaryOpen(user->comm, fname, FILE_MODE_WRITE, &viewer);
	    VecView(X,viewer);
	    PetscViewerDestroy(&viewer);
	}

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
     *  Add buildings
     * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
    if (add_building) {
        Add_Buildings(user);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
     *  Set sub domains
     * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
    //VecCreateMPI(user->comm, PetscInt n, PETSC_DETERMINE, user->subdomain);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
  	 *  Create timestepping solver context
  	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
	TSCreate(user->comm, &ts);
	TSSetProblemType(ts, TS_NONLINEAR);
	TSSetType(ts, TSEULER);
	TSSetDM(ts, user->da);
	TSSetRHSFunction(ts, R, RHSFunction, user);
	TSSetMaxTime(ts, max_time);
	TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);
	TSSetSolution(ts, X);
	TSSetTimeStep(ts, user->dt);

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
  	 *  Sets various TS parameters from user options
  	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
	TSSetFromOptions(ts); 

	/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  *
  	 *  Solver nonlinear system
  	 * - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  */
	TSSolve(ts, X);

    VecDestroy(&X);
	VecDestroy(&R);
 	VecDestroy(&user->F);
 	VecDestroy(&user->G);
 	VecDestroy(&user->B);
  	DMDestroy(&da); 
  	PetscCall(PetscFinalize());
  	return 0;
}