function generate_mesh_dam_break(dx,plot_mesh)

%dx = 1; dy = dx; % dx = dy = 100, 10, 5, 2, 1
dy = dx;
switch nargin
    case 1
        plot_mesh = 0;
end


% Make adding the path of petsc in Matlab
%addpath('YOUR_PATH_TO_PETSC/share/petsc/matlab/');
%addpath('/Users/xudo627/developments/petsc/share/petsc/matlab/');
%addpath('/Users/xudo627/donghui/CODE/dengwirda-inpoly-355c57c/');

if ~exist('inpoly')
    error('Did not find the function inpoly() in your path. Please use addpath to add directory containing inpoly().')
end

if ~exist('PetscBinaryWrite')
    error("Did not find the function PetscBinaryWrite() in your path. Please use addpath(<petsc-dir>/share/petsc/matlab).")
end

% Define the length in x and y
Lx = 1000; Ly = 500;

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
loc_cells_in  = find(in1 + in2 == 0);

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

% for each cell that is present, determine the new vertex ids
[idx_jj,idx_ii] = ind2sub([ny,nx],loc_cells_in);
connect1 = zeros(length(idx_jj),4);
for kk = 1:length(idx_jj)
    connect1(kk,:) = [new_coord_ids(idx_jj(kk),idx_ii(kk)) new_coord_ids(idx_jj(kk),idx_ii(kk)+1) new_coord_ids(idx_jj(kk)+1,idx_ii(kk)+1) new_coord_ids(idx_jj(kk)+1,idx_ii(kk))];
end



in1 = inpoly2([coordx coordy],[xb1' yb1']);
in2 = inpoly2([coordx coordy],[xb2' yb2']);
coordx(in1 | in2) = [];
coordy(in1 | in2) = [];

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
        %     in = inpoly2([coordx coordy],[xv(:,i) yv(:,i)]);
        %     %connect1(i,:) = find(in == 1);
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
end

coordz = zeros(length(coordx),1);

% Write the exodus file
block(1).connect = connect1;
exo_fname = ['DamBreak_grid' num2str(Ly/dy) 'x' num2str(Lx/dx) '_v2.exo'];
disp(['Exodus file: ' exo_fname])
write_exodus_file(exo_fname,coordx, coordy, coordz, block, []);


% Write the initial condition file
hu = 10; % upstream depth [m]
hd = 5;  % downstream depth [m]
X = zeros(size(connect1,1),3);
h = zeros(size(connect1,1),1);
x = nanmean(coordx(connect1),2);
h(x <= 400) = hu;
h(x >= 400) = hd;

if (plot_mesh)
    figure;
    patch(coordx(connect1)',coordy(connect1)',h,'LineStyle','none'); hold on; colorbar;
    xlim([0 1000]); ylim([0 500]);
    fill([400 600 600 400],[0   0   200 200],[0.5 0.5 0.5],'EdgeColor','none');
    fill([400 600 600 400],[400 400 500 500],[0.5 0.5 0.5],'EdgeColor','none');

    colormap(jet); clim([5 10]);
end

X(:,1) = h; X = X';
X = X(:);

ic_fname = ['DamBreak_grid' num2str(Ly/dy) 'x' num2str(Lx/dx) '_wetdownstream.IC'];
disp(['IC file: ' ic_fname])
PetscBinaryWrite(ic_fname,X);
