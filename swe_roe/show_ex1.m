clear;close all;clc

addpath('/Users/xudo627/developments/petsc/share/petsc/matlab/');

Nt  = length(dir('outputs/ex1*.dat')) - 1;
dof = 3;
Nx  = 200;
Ny  = 200;
dt  = 0.04;

x = 1 : 200;
y = 1 : 200;
[x,y] = meshgrid(x,y);

IC  = PetscBinaryRead('outputs/ex1_IC.dat');
IC  = reshape(IC,  [dof, length(IC)/dof]);
h0  = IC(1,:);
h0  = reshape(h0,[Nx Ny]);

figure;
imagesc(h0); colorbar; caxis([0 1]); colormap(jet);
title('t = 0s','FontSize',15);

imAlpha = ones(size(h0));
imAlpha(1:30,96:105) = 0;
imAlpha(106:200,96:105) = 0;
for i = 1 : Nt
    data  = PetscBinaryRead(['outputs/ex1_' num2str(i) '.dat']);
    data  = reshape(data,  [dof, length(data)/dof]);
    h     = data(1,:); h     = reshape(h,[Nx Ny]);
    u     = data(2,:); u     = reshape(u,[Nx Ny]);
    v     = data(3,:); v     = reshape(v,[Nx Ny]);
    pause(0.01);
    imagesc(h,'AlphaData',imAlpha); colorbar; caxis([0 10]);
    title(['t = ' num2str(i*dt) 's'],'FontSize',15);
end

u(1:31,95:106) = 0;
u(105:200,95:106) = 0;

v(1:31,95:106) = 0;
v(105:200,95:106) = 0;
figure;quiver(x,y,flipud(v),flipud(u)); hold on;
fill([95 95 106 106 95],[1 96 96 1 1]        ,[0.5 0.5 0.5],'EdgeColor','none');
fill([95 95 106 106 95],[170 200 200 170 170],[0.5 0.5 0.5],'EdgeColor','none');
xlim([1 200]);ylim([1 200]);

figure;
contour(x,y,flipud(sqrt(u.^2 + v.^2)));
colorbar; colormap(jet);


% h   = NaN(length(h0),Nt+1);
% h(:,1) = h0;

% for i = 1 : Nt - 1
%     filename = ['./Output/solution_' num2str(i) '.dat'];
%     X = PetscBinaryRead(filename);
%     X = reshape(X,   [3, length(X)/3]);
%     h(:,i+1) = X(1,:);
% end
% 
% coord = PetscBinaryRead('./Output/coord.dat');
% tri   = PetscBinaryRead('./Output/triangle.dat');
% coord = reshape(coord,[2, length(coord)/2]);
% x = coord(1,:); y = coord(2,:);
% x = x'; y = y';
% tri = reshape(tri, [3, length(tri)/3]) + 1; % node ID offset = 1
% tri = tri';
% 
% fig = figure(1);
% 
% for i = 1 : Nt 
%     filltri(h(:,i),tri,x,y,fig);
%     %colormap(jet);
%     caxis([0.1 .5]);
% end
% %filltri(h,tri,x,y);
% 
% %Check the mass balance
% 
% % fprintf(['mass at IC is: ' num2str(sum(h0)) '\n']);
% % fprintf(['mass at end is: ' num2str(sum(h)) '\n']);