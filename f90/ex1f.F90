program ex1
#include <petsc/finclude/petscts.h>
#include <petsc/finclude/petsc.h>
#include <petsc/finclude/petscdm.h>
#include <petsc/finclude/petscdmda.h>

  use petscts
  use petsc
  use petscdmda
  use petscvec
  use petscmat

  implicit none
  type AppCtx
    MPI_Comm  :: comm
    DM        :: da
    PetscInt  :: Nx, Ny
    PetscReal :: dt, hx, hy
    PetscInt  :: dof
    Vec       :: F, G, B
    PetscInt  :: xs, ys, xm, ym, xe, ye
    PetscInt  :: gxs, gxm, gys, gym, gxe, gye
  end type AppCtx

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  !  Main Program
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  TS                :: ts 
  Vec               :: X, R
  Mat               :: h ! water depth [m]
  DM                :: da
  type(AppCtx)      :: user
  PetscErrorCode    :: ierr
  PetscReal         :: max_time
  PetscInt          :: Nt, size, rank, i
  PetscBool         :: add_building
  PetscViewer       :: viewer
  character(len=80) :: outputfile
  !PetscScalar, pointer :: xarray(:,:,:)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  !  Initialize program and set problem parameters
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  if (ierr /= 0) then
    print*,'Unable to initialize PETSc'
    stop
  endif
  
  user%dt   = 0.1d0
  Nt        = 50
  user%dof  = 3 ! h, uh, vh
  user%comm = PETSC_COMM_WORLD

  call MPI_Comm_rank(user%comm, rank, ierr)
  call MPI_Comm_size(user%comm, size, ierr)
  if (rank == 0) then
    print *, 'This example is running on ', size, ' processors!!!'
  endif
  
  add_building = PETSC_FALSE
  ! Default number of cells (box), cellsize in x-direction, and y-direction
  user%Nx   = 51
  user%hx   = 1.0d0
  user%hy   = 1.0d0

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-Nx',       &
                          user%Nx,PETSC_NULL_BOOL,ierr)
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-Ny',       &
                          user%Ny,PETSC_NULL_BOOL,ierr)
  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-Nt',       &
                          Nt,PETSC_NULL_BOOL,ierr)
  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-hx',      &
                           user%hx,PETSC_NULL_BOOL,ierr)
  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-hy',      &
                           user%hy,PETSC_NULL_BOOL,ierr)
  call PetscOptionsGetReal(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-dt',      &
                           user%dt,PETSC_NULL_BOOL,ierr)
  call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-b',       &
                           add_building,PETSC_NULL_BOOL,ierr)
  
  max_time = real(Nt)*user%dt
  user%Ny  = user%Nx

  ! Default outputfile name
  write(outputfile, '(A15,I3,A4)')  './ex1_', Nt, '.dat'
  outputfile = trim(outputfile)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  !  Initialize program and set problem parameters
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  call DMDACreate2d(user%comm, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,       &
       DMDA_STENCIL_BOX, user%Nx, user%Ny, PETSC_DECIDE, PETSC_DECIDE,         &
       user%dof, 1, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, da, ierr)
  call DMSetFromOptions(da, ierr)
  call DMSetUp(da, ierr)
  user%da = da 
  
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  !  Read buildings to domain
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  call GetBuilding(user, add_building, ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  !  Extract global vectors from DMDA
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  call DMCreateGlobalVector(da, X, ierr)
  call VecDuplicate(X, user%F, ierr)
  call VecDuplicate(X, user%G, ierr)
  call VecDuplicate(X, user%B, ierr)
  call VecDuplicate(X, R, ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  !  Initial Condition
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  call FormInitialSolution(X, user, ierr)
  call PetscViewerBinaryOpen(user%comm,'./ex1_IC.dat',FILE_MODE_WRITE,viewer,ierr)
  call VecView(X,viewer,ierr)
  !call MatView(h, viewer, ierr)
  call PetscViewerDestroy(viewer,ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  !  Create timestepping solver context
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !

  call TSCreate(user%comm, ts, ierr)
  call TSSetProblemType(ts, TS_NONLINEAR, ierr)
  call TSSetType(ts, TSEULER, ierr)
  call TSSetDM(ts, da, ierr)
  call TSSetRHSFunction(ts, R, RHSFunction, user, ierr)
  call TSSetMaxTime(ts, max_time, ierr)
  call TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER, ierr)

  call TSSetSolution(ts, X, ierr)
  call TSSetTimeStep(ts, user%dt, ierr)
  
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  ! Sets various TS parameters from user options
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  call TSSetFromOptions(ts, ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  ! Solver nonlinear system
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  call TSSolve(ts, X, ierr); CHKERRA(ierr)

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  ! Save the solution to binary
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  !call PetscViewerBinaryOpen(user%comm,outputfile,FILE_MODE_WRITE,viewer,ierr)
  !call VecView(X,viewer,ierr)
  !call PetscViewerDestroy(viewer,ierr)
  
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  ! Free work space
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
  call VecDestroy(X, ierr)
  call VecDestroy(R, ierr)
  call VecDestroy(user%F, ierr)
  call VecDestroy(user%G, ierr)
  call VecDestroy(user%B, ierr)
  call TSDestroy(ts, ierr)
  call DMDestroy(da, ierr)

  call PetscFinalize(ierr)

contains
  
  subroutine FormInitialSolution(X, user, ierr)

    implicit none
    Vec            :: X
    type(AppCtx)   :: user
    PetscErrorCode :: ierr

    DM             :: da
    PetscInt       :: i, j, k, xs, ys, xm, ym, Nx, Ny
    PetscInt       :: gxs, gys, gxm, gym

    !real, allocatable    :: F, G
    PetscScalar, pointer :: xarray(:,:,:), farray(:,:,:), garray(:,:,:)

    da = user%da
    Nx = user%Nx
    Ny = user%Ny

    !allocate(user%F(Ny,Nx+1,3), user%G(Ny+1,Nx,3))

    ! Get pointer to vector data
    call DMDAVecGetArrayF90(da, X, xarray, ierr)
    call DMDAVecGetArrayF90(da, user%F, farray, ierr)
    call DMDAVecGetArrayF90(da, user%G, garray, ierr)

    ! Get local grid boundaries
    call DMDAGetCorners(da, xs, ys, PETSC_NULL_INTEGER,                        &
                            xm, ym, PETSC_NULL_INTEGER, ierr)
    call DMDAGetGhostCorners(da,gxs,gys,PETSC_NULL_INTEGER,gxm,gym,PETSC_NULL_INTEGER,ierr)
    
    !print *, 'xs = ', xs, 'xm = ', xm
    !print *, 'ys = ', ys, 'ym = ', ym

    !print *, 'gxs = ', gxs, 'gxm = ', gxm
    !print *, 'gys = ', gys, 'gym = ', gym

    ! k = 0: h, k = 1: uh, k = 2: vh
    do k = 0, user%dof - 1
      do j = ys, ys + ym - 1
        do i = xs, xs + xm - 1

          if ( k == 0 ) then
            if ( j >= Ny - 10 .and. i < 10 ) then ! higher water in the SW 
              xarray(k,i,j) = 1.0d0
            else 
              xarray(k,i,j) = 0.1d0
            endif
          else
            xarray(k,i,j) = 0.0d0 ! initialize uh and vh with 0
          endif

          ! initialize the flux array with 0
          farray(k,i,j) = 0.0d0
          garray(k,i,j) = 0.0d0
          
          !if (i == 1 .and. j == 41) then
          !  print *, 'Initial Condition: h(2, 42) = ', xarray(0, i, j)
          !endif

        enddo
      enddo
    enddo

    ! Restore vectors
    call DMDAVecRestoreArrayF90(da, X, xarray, ierr)
    call DMDAVecRestoreArrayF90(da, user%F, farray, ierr)
    call DMDAVecRestoreArrayF90(da, user%G, garray, ierr)

    !user%F = F
    !user%G = G

    !deallocate(F, G)
    print *, 'Initialization done!!!'
  end subroutine FormInitialSolution

  subroutine RHSFunction(ts, t, X, F, user, ierr)
    
    use petscts
    use petscdmda
      
    implicit none
    TS             :: ts
    PetscReal      :: t
    Vec            :: X, F
    type(AppCtx)   :: user
    PetscErrorCode :: ierr

    DM             :: da
    PetscInt       :: i, j, k, xs, ys, xm, ym, Nx, Ny
    PetscInt       :: gxs, gys, gxm, gym, rank
    Vec            :: localX, localF, localG
    PetscScalar, pointer :: xarray(:,:,:), f1(:,:,:), farray(:,:,:), garray(:,:,:)
    
    ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !
    !  corrector
    ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - !

    da = user%da
    Nx = user%Nx
    Ny = user%Ny

    call DMGetLocalVector(da, localX, ierr)
    call DMGetLocalVector(da, localF, ierr)
    call DMGetLocalVector(da, localG, ierr)

    !
    ! Scatter ghost points to local vector, using the 2-step process
    ! DMGlobalToLocalBegin(), DMGlobalToLocalEnd()
    ! By placing code between these tow statements, computations can be
    ! done while messages are in transition
    !
    call DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX, ierr)
    call DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX, ierr)

    call DMGlobalToLocalBegin(da, user%F, INSERT_VALUES, localF, ierr)
    call DMGlobalToLocalEnd(da, user%F, INSERT_VALUES, localF, ierr)

    call DMGlobalToLocalBegin(da, user%G, INSERT_VALUES, localG, ierr)
    call DMGlobalToLocalEnd(da, user%G, INSERT_VALUES, localG, ierr)

    ! Get pointers to vector data
    call DMDAVecGetArrayF90(da,localX, xarray, ierr)
    call DMDAVecGetArrayF90(da,localF, farray, ierr)
    call DMDAVecGetArrayF90(da,localG, garray, ierr)
    call DMDAVecGetArrayF90(da, F, f1, ierr)

    ! Get local grid boundaries 
    call DMDAGetCorners(da, xs, ys, PETSC_NULL_INTEGER,                        &
                            xm, ym, PETSC_NULL_INTEGER, ierr)
    call DMDAGetGhostCorners(da, gxs, gys, PETSC_NULL_INTEGER,                 &
                                 gxm, gym, PETSC_NULL_INTEGER,ierr)
    user%xs = xs
    user%ys = ys
    user%xm = xm
    user%ym = ym

    user%gxs = gxs
    user%gys = gys
    user%gxm = gxm
    user%gym = gym

    user%xe = user%xs + user%xm - 1
    user%ye = user%ys + user%ym - 1
    user%gxe = user%gxs + user%gxm - 1
    user%gye = user%gys + user%gym - 1
    
    call MPI_Comm_rank(user%comm, rank, ierr)

    call fluxes(xarray, farray, garray, user, ierr)

    !print *, 'fluxes is done in rank ', rank, xs, xm,ys,ym

    do j = ys, ys + ym - 1
      do i = xs, xs + xm - 1
        do k = 0, user%dof - 1
          ! U(j,k,l)=U(j,k,l)-dt2*(F(j+1,k,l)-F(j,k,l)+G(j,k+1,l)-G(j,k,l));
          f1(k,i,j) = -(farray(k,i+1,j) - farray(k,i,j)  &
                      + garray(k,i,j+1) - garray(k,i,j))
        enddo
      enddo
    enddo
    
    ! Restore vectors
    call DMDAVecRestoreArrayF90(da, localX, xarray, ierr)

    call DMDAVecRestoreArrayF90(da, localF, farray, ierr)

    call DMDAVecRestoreArrayF90(da, localG, garray, ierr)

    call DMDAVecRestoreArrayF90(da, F, f1, ierr)

    call DMRestoreLocalVector(da, localX, ierr)
    call DMRestoreLocalVector(da, localF, ierr)
    call DMRestoreLocalVector(da, localG, ierr)


  end subroutine RHSFunction

  subroutine fluxes(xarray,farray, garray, user, ierr)

    implicit none
    type(AppCtx)   :: user
    PetscErrorCode :: ierr

    DM             :: da
    Vec            :: localB
    PetscReal      :: sn, cn, a, amax, hl, hr, ul, ur, vl, vr
    PetscInt       :: i, j,Nx, Ny
    integer, save  :: tstep = 0
    PetscScalar    :: xarray(0:2,user%gxs:user%gxe,user%gys:user%gye)
    PetscScalar    :: farray(0:2,user%gxs:user%gxe,user%gys:user%gye)
    PetscScalar    :: garray(0:2,user%gxs:user%gxe,user%gys:user%gye)
    PetscScalar, pointer :: barray(:,:,:)

    da = user%da
    Nx = user%Nx
    Ny = user%Ny
    tstep = tstep + 1

    call DMGetLocalVector(da, localB, ierr)
    call DMGlobalToLocalBegin(da, user%B, INSERT_VALUES, localB, ierr)
    call DMGlobalToLocalEnd(da, user%B, INSERT_VALUES, localB, ierr)
    call DMDAVecGetArrayF90(da, localB, barray, ierr)
        
    amax = 0.0d0

    do i = user%gxs+1, user%gxs + user%gxm - 1
      do j = user%gys+1, user%gys + user%gym - 1

        ! - - - - - - - - - - - - - - - !
        ! Compute fluxes in x-driection !
        ! - - - - - - - - - - - - - - - !

        sn = 0.0d0
        cn = 1.0d0
        if (i == 0) then      ! Enforce wall boundary on left side of box (west)
          hr = xarray(0, i, j)
          ur = xarray(1, i, j)/hr
          vr = xarray(2, i, j)/hr
          call solver(hr, hr, -ur, ur, vr, vr, sn, cn, farray(:, i, j), a)
          amax = max(a, amax)
        else if (i == Nx) then
          hl = xarray(0, i-1, j)
          ul = xarray(1, i-1, j)/hl
          vl = xarray(2, i-1, j)/hl
          call solver(hl, hl, ul, -ul, vl, vl, sn, cn, farray(:, i, j), a)
          amax = max(a, amax)
        else if (barray(0, i, j) == 1.0d0 .and. barray(0, i-1, j) == 0.0d0) then
          hl = xarray(0, i-1, j)
          ul = xarray(1, i-1, j)/hl
          vl = xarray(2, i-1, j)/hl
          call solver(hl, hl, ul, -ul, vl, vl, sn, cn, farray(:, i, j), a)
          amax = max(a, amax)
        else if (barray(0, i, j) == 0.0d0 .and. barray(0, i-1, j) == 1.0d0) then
          hr = xarray(0, i, j)
          ur = xarray(1, i, j)/hr
          vr = xarray(2, i, j)/hr
          call solver(hr, hr, -ur, ur, vr, vr, sn, cn, farray(:, i, j), a)
          amax = max(a, amax)
        else if (barray(0, i, j) == 1.0d0 .and. barray(0, i-1, j) == 1.0d0) then
          farray(:, i, j) = (/0, 0, 0/)
        else
          hl = xarray(0, i-1, j)
          hr = xarray(0, i, j)
          ul = xarray(1, i-1, j)/hl
          ur = xarray(1, i, j)/hr
          vl = xarray(2, i-1, j)/hl
          vr = xarray(2, i, j)/hr
          call solver(hl, hr, ul, ur, vl, vr, sn, cn, farray(:, i, j), a)
          amax = max(a, amax)
        endif

        ! - - - - - - - - - - - - - - - !
        ! Compute fluxes in y-driection !
        ! - - - - - - - - - - - - - - - !

        sn = 1.0d0
        cn = 0.0d0
        if (j == 0) then      ! Enforce wall boundary on bottom of box (south)
          hr = xarray(0, i, j)
          ur = xarray(1, i, j)/hr
          vr = xarray(2, i, j)/hr
          call solver(hr, hr, ur, ur, -vr, vr, sn, cn, garray(:, i, j), a)
          amax = max(a, amax)
        else if (j == Ny) then
          hl = xarray(0, i, j-1)
          ul = xarray(1, i, j-1)/hl
          vl = xarray(2, i, j-1)/hl
          call solver(hl, hl, ul, ul, vl, -vl, sn, cn, garray(:, i, j), a)
          amax = max(a, amax)
        else if (barray(0, i, j) == 1.0d0 .and. barray(0, i, j-1) == 0.0d0) then
          hl = xarray(0, i, j-1)
          ul = xarray(1, i, j-1)/hl
          vl = xarray(2, i, j-1)/hl
          call solver(hl, hl, ul, ul, vl, -vl, sn, cn, garray(:, i, j), a)
          amax = max(a, amax)
        else if (barray(0, i, j) == 0.0d0 .and. barray(0, i, j-1) == 1.0d0) then
          hr = xarray(0, i, j)
          ur = xarray(1, i, j)/hr
          vr = xarray(2, i, j)/hr
          call solver(hr, hr, ur, ur, -vr, vr, sn, cn, garray(:, i, j), a)
        else if (barray(0, i, j) == 1.0d0 .and. barray(0, i, j-1) == 1.0d0) then
          garray(:, i, j) = (/0,0,0/)
        else
          hl = xarray(0, i, j-1)
          hr = xarray(0, i, j)
          ul = xarray(1, i, j-1)/hl
          ur = xarray(1, i, j)/hr
          vl = xarray(2, i, j-1)/hl
          vr = xarray(2, i, j)/hr
          call solver(hl, hr, ul, ur, vl, vr, sn, cn, garray(:, i, j), a)
          amax = max(a, amax)
        endif

        if (i == 1 .and. j == 41) then ! to compare with matlab
          print "(a10,i2,a1,i2,a5,f5.3,a5,f6.3,a5,f6.3)", 'Depth at (',i+1,',',j+1,') is ',    &
                                          xarray(0,i,j), 'fij=', farray(0,i,j), 'gij=', garray(0,i,j)
          print *,'gij=',garray(0,i,j)
        endif
        !if (i == 1) then
        !  print *, 'ghost cell depth at ', i+1, user%gys+user%gym,'is ',       &
        !            xarray(0,i, user%gys+user%gym-1)
        !endif

      enddo
    enddo

    !fprintf('Time Step = %d, Courant Number = %g \n',tstep,amax*dt2*2)
    write (*, fmt = '(a,i3,a,f5.3)') 'Time Step = ', tstep, ', Courant Number = ', amax*user%dt*2

    call DMDAVecRestoreArrayReadF90(da, localB, barray, ierr)
    call DMRestoreLocalVector(da, localB, ierr)

  end subroutine fluxes
  
  subroutine solver(hl, hr, ul, ur, vl, vr, sn, cn, F, amax)
    implicit none
    PetscReal, intent(in)  :: hl, hr, ul, ur, vl, vr, sn, cn
    PetscReal, intent(out) :: F(3), amax
    PetscReal :: duml, dumr, cl, cr, hhat, uhat, vhat, chat, uperp, dh, du, dv
    PetscReal :: duperp, dW(3), al1, al3, ar1, ar3, R(3,3), da1, da3, a1, a2, a3
    PetscReal :: dupar, uperpl, uperpr, A(3,3), FL(3), FR(3)
    PetscReal :: grav
    PetscInt  :: i, j

    grav = 9.806d0

    ! Compute Roe averages
    duml   = hl**0.5d0
    dumr   = hr**0.5d0
    cl     = (grav*hl)**0.5d0
    cr     = (grav*hr)**0.5d0
    hhat   = duml*dumr
    uhat   = (duml*ul + dumr*ur)/(duml + dumr)
    vhat   = (duml*vl + dumr*vr)/(duml + dumr)
    chat   = (0.5d0*grav*(hl + hr))**0.5d0
    uperp  = uhat*cn + vhat*sn
    dh     = hr - hl
    du     = ur - ul
    dv     = vr - vl
    dupar  = -du*sn + dv*cn
    duperp = du*cn + dv*sn
    dW(1)  = 0.5d0*(dh - hhat*duperp/chat)
    dW(2)  = hhat*dupar
    dW(3)  = 0.5d0*(dh + hhat*duperp/chat)

    uperpl = ul*cn + vl*sn
    uperpr = ur*cn + vr*sn
    al1 = uperpl - cl
    al3 = uperpl + cl
    ar1 = uperpr - cr
    ar3 = uperpr + cr
    R(1,1) = 1.0d0
    R(1,2) = 0.0d0
    R(1,3) = 1.0d0
    R(2,1) = uhat - chat*cn
    R(2,2) = -sn
    R(2,3) = uhat + chat*cn
    R(3,1) = vhat - chat*sn
    R(3,2) = cn
    R(3,3) = vhat + chat*sn

    da1 = max(0.0d0, 2.0d0*(ar1 - al1))
    da3 = max(0.0d0, 2.0d0*(ar3 - al3))
    a1  = abs(uperp - chat)
    a2  = abs(uperp)
    a3  = abs(uperp + chat)

    ! Critical flow fix
    if ( a1 < da1 ) then
      a1 = 0.5d0*(a1*a1/da1 + da1)
    endif
    if ( a3 < da3 ) then
      a3 = 0.5d0*(a3*a3/da3 + da3)
    endif

    ! Compute interface flux
    do i = 1, 3
      do j = 1, 3
        A(i,j) = 0.0d0
      enddo
    enddo
    A(1,1) = a1
    A(2,2) = a2
    A(3,3) = a3
    FL     = (/uperpl*hl, &
               ul*uperpl*hl + 0.5d0*grav*hl*hl*cn, &
               vl*uperpl*hl + 0.5d0*grav*hl*hl*sn/)
    FR     = (/uperpr*hr, &
               ur*uperpr*hr + 0.5d0*grav*hr*hr*cn, &
               vr*uperpr*hr + 0.5d0*grav*hr*hr*sn/)

    F = 0.5d0*(FL + FR - matmul(R,matmul(A,dW)))
    amax = chat + abs(uperp)

  end subroutine solver

  subroutine WriteMat(filename, M)
    implicit none
    character(len = 42) :: filename
    PetscReal, dimension(:,:) :: M
    integer              :: i, j, s(2)
    
    s = shape(M)
    open(2, file = filename)
    do j = 1, s(1)
      do i = 1, s(2)
        write (2, '(F4.2,XX)', advance = 'no') M(j,i)
      enddo
      write(2,*)
    enddo 

    close(2)

  end subroutine WriteMat

  subroutine GetBuilding(user, add_building, ierr)
    implicit none
    type(AppCtx)   :: user
    PetscBool      :: add_building
    PetscErrorCode :: ierr

    PetscInt       :: i, j, k, xs, ys, xm, ym
    PetscScalar, pointer   :: A(:,:,:)
    character *20  :: filename
    PetscBool      :: flg
    
    write (*, fmt = '(a)', advance = 'no') 'Initializing the building cells: '

    if ( add_building ) then
      call PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-s', &
                                 filename, flg, ierr)
      if (.not. flg) then
        print *, 'Please specify the name of the file!!!'
      endif

      call ReadMat(user%da, user%B, filename, user%Nx, user%Ny, ierr)

    else
      ! - - - - - - - - - - - - - - - - - - - - - - - - - - - !
      ! No buildings in the domain
      ! - - - - - - - - - - - - - - - - - - - - - - - - - - - !
      write (*, fmt = '(a)') 'No buildings in the domain '

      call DMCreateGlobalVector(user%da, user%B, ierr)
      call DMDAVecGetArrayF90(user%da, user%B, A, ierr)
      call DMDAGetCorners(user%da, xs, ys, PETSC_NULL_INTEGER,                 &
                                   xm, ym, PETSC_NULL_INTEGER, ierr)

      do k = 0, user%dof - 1
        do j = ys, ys + ym - 1
          do i = xs, xs + xm - 1

            A(k, i, j) = 0.0d0

          enddo
        enddo
      enddo

      call DMDAVecRestoreArrayF90(user%da, user%B, A, ierr)

    endif

    write (*, fmt = '(a)') '--> Successful!!!'

  end subroutine GetBuilding

  subroutine ReadMat(da, V, filename, m, n, ierr)
    use petscmat
    implicit none
    DM             :: da
    Vec            :: V
    character *20  :: filename
    PetscInt       :: m, n 
    PetscErrorCode :: ierr
    
    PetscInt       :: i, j, k, xs, ys, xm, ym
    integer        :: s(2)
    PetscScalar, pointer   :: A(:,:,:)
    PetscReal, allocatable :: mat(:,:) 
    
    allocate( mat(m, n) )
    print *, 'Reading ', filename
    open(12, file = filename)
    read(12, *) mat
    close(12)
    mat = transpose(mat)
    s   = shape(mat)
    if (m /= s(1) .or. n /= s(2)) then
      stop 'size of the file does not match'
    endif

    call DMCreateGlobalVector(da, V, ierr)
    call DMDAVecGetArrayF90(da, V, A, ierr)
    call DMDAGetCorners(da, xs, ys, PETSC_NULL_INTEGER,                        &
                            xm, ym, PETSC_NULL_INTEGER, ierr)
    
    do k = 0, user%dof - 1
      do j = ys, ys + ym - 1
        do i = xs, xs + xm - 1

          if (k == 0) then
            A(k, i, j) = mat(j+1, i+1)
          else
            A(k, i, j) = 0.0d0
          endif

        enddo
      enddo
    enddo
    
    call DMDAVecRestoreArrayF90(da, V, A, ierr)

    deallocate( mat )

  end subroutine ReadMat

  subroutine Vec2Mat(V, A, num_of_dof, user, ierr)
    implicit none
    Vec            :: V, localV
    Mat            :: A
    PetscInt       :: num_of_dof
    type(AppCtx)   :: user
    PetscErrorCode :: ierr

    PetscScalar, pointer :: varray(:,:,:)
    PetscInt             :: i, j, xs, ys, xm, ym, mstart, mend

    call DMGetLocalVector(user%da, localV, ierr)

    call DMGlobalToLocalBegin(user%da, V, INSERT_VALUES, localV, ierr)
    call DMGlobalToLocalEnd(user%da, V, INSERT_VALUES, localV, ierr)

    ! Get pointers to vector data
    call DMDAVecGetArrayF90(user%da,localV, varray, ierr)

    ! Get local grid boundaries 
    call DMDAGetCorners(user%da, xs, ys, PETSC_NULL_INTEGER,                   &
                                 xm, ym, PETSC_NULL_INTEGER, ierr)
    ! 
    call MatCreate(user%comm, A, ierr)
    call MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, user%Nx, user%Ny, ierr)
    call MatSetFromOptions(A, ierr)
    call MatSetUp(A, ierr)

    do i = xs, xs + xm - 1
      do j = ys, ys + ym - 1
        call MatSetValues(A,1,j,1,i,varray(num_of_dof,i,j),INSERT_VALUES,ierr)
      enddo
    enddo

    call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr)
    call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr)
    
    ! Restore vectors
    call DMDAVecRestoreArrayF90(user%da, localV, varray, ierr)
    call DMRestoreLocalVector(user%da, localV, ierr)


  end subroutine Vec2Mat

end program ex1