function generate_mesh_dam_break()

clear;close all;clc;

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

dx = 1; dy = 1; % dx = dy = 100, 10, 5, 2, 1

% Define the lenght in x and y 
Lx = 1000; Ly = 500;

% Node coordinates
coordx = 0 : dx : Lx;
coordy = 0 : dy : Ly;
[coordx,coordy] = meshgrid(coordx,coordy);
coordx = coordx(:);
coordy = coordy(:);

% Cell center coordinates
xc = dx/2 : dx : Lx-dx/2;
yc = dy/2 : dy : Ly-dy/2;
[xc,yc] = meshgrid(xc,yc);
xc = xc(:);
yc = yc(:);

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
in1 = inpoly2([coordx coordy],[xb1' yb1']);
in2 = inpoly2([coordx coordy],[xb2' yb2']);
coordx(in1 | in2) = [];
coordy(in1 | in2) = [];

xv   = NaN(4,length(xc)); yv   = NaN(4,length(xc));
xv(1,:,:) = xc - dx/2; xv(2,:,:) = xc + dx/2; xv(3,:,:) = xc + dx/2; xv(4,:,:) = xc - dx/2;
yv(1,:,:) = yc - dy/2; yv(2,:,:) = yc - dy/2; yv(3,:,:) = yc + dy/2; yv(4,:,:) = yc + dy/2;

coordx = round(coordx,5);
coordy = round(coordy,5);
xv = round(xv,5);
yv = round(yv,5);
figure;
plot(coordx,coordy,'ro','LineWidth',2);hold on; axis equal;

connect1 = NaN(size(xv,2),4);
for i = 1 : size(xv,2)
    disp([num2str(i) '/' num2str(size(xv,2))]);
%     in = inpoly2([coordx coordy],[xv(:,i) yv(:,i)]);
%     %connect1(i,:) = find(in == 1);
    for j = 1 : 4
        connect1(i,j) = find(coordx == xv(j,i) & coordy == yv(j,i));  
    end
end
for i = 1 : size(connect1,1)
    iver = [connect1(i,:) connect1(i,1)];
    plot(coordx(iver),coordy(iver),'b-','LineWidth',1);
end
coordz = zeros(length(coordx),1);

% Write the exodus file
block(1).connect = connect1;
write_exodus_file(['DamBreak_grid' num2str(Ly/dy) 'x' num2str(Lx/dx) '_v2.exo'],coordx, coordy, coordz, block, []);


% Write the initial condition file
hu = 10; % upstream depth [m]
hd = 5;  % downstream depth [m]
X = zeros(size(connect1,1),3);
h = zeros(size(connect1,1),1); 
x = nanmean(coordx(connect1),2);
h(x <= 400) = hu;
h(x >= 400) = hd;

figure;
patch(coordx(connect1)',coordy(connect1)',h,'LineStyle','none'); hold on; colorbar;
xlim([0 1000]); ylim([0 500]);
fill([400 600 600 400],[0   0   200 200],[0.5 0.5 0.5],'EdgeColor','none');
fill([400 600 600 400],[400 400 500 500],[0.5 0.5 0.5],'EdgeColor','none');

colormap(jet); clim([5 10]);
X(:,1) = h; X = X';
X = X(:);

PetscBinaryWrite(['../initial_conditions/DamBreak_grid' num2str(Ly/dy) 'x' num2str(Lx/dx) '_wetdownstream.IC'],X);
