function generate_mesh(dx,plot_mesh)

% dx = dy = 0.05, 0.1, 0.25, 0.5, 1
dy = dx;
switch nargin
    case 1
        plot_mesh = 0;
end


% Make adding the path of petsc in Matlab
%addpath('YOUR_PATH_TO_PETSC/share/petsc/matlab/');
addpath('/qfs/people/feng779/RDycore/petsc/share/petsc/matlab/');
addpath('/qfs/people/feng779/RDycore/matlat_lib/inpoly/');
addpath('/qfs/people/feng779/RDycore/toy-problems/share/meshes')

if ~exist('inpoly')
    error('Did not find the function inpoly() in your path. Please use addpath to add directory containing inpoly().')
end

if ~exist('PetscBinaryWrite')
    error("Did not find the function PetscBinaryWrite() in your path. Please use addpath(<petsc-dir>/share/petsc/matlab).")
end

% Define the length in x and y
Lx = 5; Ly = 5;

% Node coordinates
coordx = 0 : dx : Lx;
coordy = 0 : dy : Ly;
[coordx,coordy] = meshgrid(coordx,coordy);


% create a matrix with vertex ids
[nvy,nvx] = size(coordx);
coord_ids = reshape([1:nvy*nvx],nvy,nvx);


coordx = coordx(:);
coordy = coordy(:);

% Cell center coordinates
xc = dx/2 : dx : Lx-dx/2;
yc = dy/2 : dy : Ly-dy/2;
[xc,yc] = meshgrid(xc,yc);
xc_2d = xc;
yc_2d = yc;

[ny,nx] = size(xc);

xc = xc(:);
yc = yc(:);

cells = zeros(nx*ny,4);
count = 0;
for ii = 1:nx
    for jj = 1:ny
        count = count + 1;
        cells(count,:) = [coord_ids(jj,ii) coord_ids(jj,ii+1) coord_ids(jj+1,ii+1) coord_ids(jj+1,ii) ];
    end
end


% remove the cells iside the dam
offset = dx/10;
xb1 = [400+offset 600-offset 600-offset 400+offset 400+offset];
yb1 = [0 0 200-offset 200-offset 0];
xb2 = [400+offset 600-offset 600-offset 400+offset 400+offset];
yb2 = [400+offset 400+offset 500 500 400+offset];
in1 = inpolygon(xc,yc,xb1',yb1');
in2 = inpolygon(xc,yc,xb2',yb2');
xc(in1 | in2) = [];
yc(in1 | in2) = [];


% determine the cells that are present afte the dam is removed
loc_cells_in  = find(in1 + in2 == 0)

% determine if a coordinate is present after the dam is removed
coord_present = coord_ids*0;
coord_present(cells(loc_cells_in,:)) = 1;
ncoords_preset = sum(sum(coord_present));

% determine new ids of vertices that are present
new_coord_ids = coord_ids*0;
loc_coords_in = find(coord_present == 1);
new_coord_ids(loc_coords_in) = [1:ncoords_preset];
%new_coordx = coordx(loc_coords_in);
%new_coordy = coordy(loc_coords_in);


%new_coords_ids = coord_ids

% for each cell that is present, determine the new vertex ids
[idx_jj,idx_ii] = ind2sub([ny,nx],loc_cells_in);
connect1 = zeros(length(idx_jj),4);
for kk = 1:length(idx_jj)
    connect1(kk,:) = [new_coord_ids(idx_jj(kk),idx_ii(kk)) new_coord_ids(idx_jj(kk),idx_ii(kk)+1) new_coord_ids(idx_jj(kk)+1,idx_ii(kk)+1) new_coord_ids(idx_jj(kk)+1,idx_ii(kk))];
end

%{
in1 = inpoly2([coordx coordy],[xb1' yb1']);
in2 = inpoly2([coordx coordy],[xb2' yb2']);
coordx(in1 | in2) = [];
coordy(in1 | in2) = [];
%}

xv   = NaN(4,length(xc)); yv   = NaN(4,length(xc));
xv(1,:,:) = xc - dx/2; xv(2,:,:) = xc + dx/2; xv(3,:,:) = xc + dx/2; xv(4,:,:) = xc - dx/2;
yv(1,:,:) = yc - dy/2; yv(2,:,:) = yc - dy/2; yv(3,:,:) = yc + dy/2; yv(4,:,:) = yc + dy/2;

% Do not roundoff the vertex coordinates and do not use the approach to
% find vertex ids using `find`
if (0)
    coordx = round(coordx,5);
    coordy = round(coordy,5);
    xv = round(xv,5);
    yv = round(yv,5);

    connect1 = NaN(size(xv,2),4);
    for i = 1 : size(xv,2)
        disp([num2str(i) '/' num2str(size(xv,2))]);
        for j = 1 : 4
            connect1(i,j) = find(coordx == xv(j,i) & coordy == yv(j,i));
        end
    end

end


if (plot_mesh)
    figure;
    plot(coordx,coordy,'ro','LineWidth',2);hold on; axis equal;
    for i = 1 : size(connect1,1)
        iver = [connect1(i,:) connect1(i,1)];
        plot(coordx(iver),coordy(iver),'b-','LineWidth',1);
    end

    mesh_fname = ['Mesh_dx' num2str(dx) '.png'];
    saveas(gcf,mesh_fname)
end

coordz = zeros(length(coordx),1);

% Write the exodus file
block(1).connect = connect1;
exo_fname = ['MMS_mesh_dx' num2str(dx) '.exo'];
disp(['Exodus file: ' exo_fname])
write_exodus_file(exo_fname,coordx, coordy, coordz, block, []);


% Write the initial condition file
hu = 10; % upstream depth [m]
hd = 5;  % downstream depth [m]
X = zeros(size(connect1,1),3);
%h = zeros(size(connect1,1),1);
%x = nanmean(coordx(connect1),2);
%h(x <= 400) = hu;
%h(x >= 400) = hd;

Lx = 5;
Ly = 5;

function h_exact = h_solution(xin, yin, tin)
   h0 = 0.005;
   t0 = 20;
   h_exact = h0 * ( 1+sin(pi*xin/Lx).*sin(pi*yin/Ly) ) * exp(tin/t0);
end

function u_exact = u_solution(xin, yin, tin)
   u0 = 0.005;
   t0 = 20;
   u_exact = u0 * ( 1+sin(pi*xin/Lx).*sin(pi*yin/Ly) ) * exp(tin/t0);
end

function v_exact = v_solution(xin, yin, tin)
   v0 = 0.005;
   t0 = 20;
   v_exact = v0 * ( 1+sin(pi*xin/Lx).*sin(pi*yin/Ly) ) * exp(tin/t0);
end

h = h_solution(xc, yc, 0)
u = u_solution(xc, yc, 0)
v = v_solution(xc, yc, 0)

size(h)


if (plot_mesh)
    figure;
    patch(coordx(connect1)',coordy(connect1)',h,'LineStyle','none'); hold on; colorbar;
    xlim([0 5]); ylim([0 5]);
    %fill([400 600 600 400],[0   0   200 200],[0.5 0.5 0.5],'EdgeColor','none');
    %fill([400 600 600 400],[400 400 500 500],[0.5 0.5 0.5],'EdgeColor','none');

    colormap(jet); caxis([0.005 0.015]);
    IC_fname = ['initial_h_dx' num2str(dx) '.png'];
    saveas(gcf,IC_fname)
end

X(:,1) = h; 
X(:,2) = h.*u;
X(:,3) = h.*v;
X = X';
X = X(:);


ic_fname = ['../initial_conditions/MMS_dx' num2str(dx) '.IC'];
disp(['IC file: ' ic_fname])
PetscBinaryWrite(ic_fname,X);

end
