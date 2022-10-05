clear;close all;clc

addpath('/Users/xudo627/developments/petsc/share/petsc/matlab/');

Nt  = 100;
dof = 3;
Nx  = 51;
Ny  = 51;
dt  = 0.1;

IC  = PetscBinaryRead('outputs/ex1_IC.dat');
IC  = reshape(IC,  [dof, length(IC)/dof]);
h0  = IC(1,:);
h0  = reshape(h0,[Nx Ny]);

figure;
imagesc(h0); colorbar; caxis([0 1]); colormap(jet);
title('t = 0s','FontSize',15);

for i = 1 : Nt
    data  = PetscBinaryRead(['outputs/ex1_' num2str(i) '.dat']);
    data  = reshape(data,  [dof, length(data)/dof]);
    h     = data(1,:);
    h     = reshape(h,[Nx Ny]);
    pause(0.1);
    imagesc(h); colorbar; caxis([0 1]);
    title(['t = ' num2str(i*dt) 's'],'FontSize',15);
end



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