clear;close all;clc;

% Make adding the path of petsc in Matlab
%addpath('YOUR_PATH_TO_PETSC/share/petsc/matlab/');

dx = 1; dy = 1;

% Define the lenght in x and y 
Lx = 10; Ly = 5;

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
offset = 0.1;
xb1 = [4+offset 6-offset 6-offset 4+offset 4+offset];
yb1 = [0 0 2-offset 2-offset 0];
xb2 = [4+offset 6-offset 6-offset 4+offset 4+offset];
yb2 = [4+offset 4+offset 5 5 4+offset];
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
 

figure;
plot(coordx,coordy,'ro','LineWidth',2);hold on; axis equal;

connect1 = NaN(size(xv,2),4);
for i = 1 : size(xv,2)
    disp([num2str(i) '/' num2str(size(xv,2))]);
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
write_exodus_file(coordx, coordy, coordz, connect1, ['DamBreakd_grid' num2str(Ly) 'x' num2str(Lx) '.exo']);

% Write the initial condition file
hu = 10; % upstream depth [m]
hd = 5;  % downstream depth [m]
X = zeros(size(connect1,1),3);
h = zeros(size(connect1,1),1); 
x = nanmean(coordx(connect1),2);
h(x <= 4) = hu;
h(x >= 4) = hd;

figure;
patch(coordx(connect1)',coordy(connect1)',h,'LineStyle','none'); hold on; colorbar;
X(:,1) = h; X = X';
X = X(:);

PetscBinaryWrite(['../initial_conditions/DamBreak_grid' num2str(Ly) 'x' num2str(Lx) '_wetdownstream.IC'],X);
