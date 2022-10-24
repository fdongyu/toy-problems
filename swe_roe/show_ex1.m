clear;close all;clc

% User need to provide petsc directory
addpath('/Users/xudo627/developments/petsc/share/petsc/matlab/');

Nt  = length(dir('outputs/ex1*.dat')) - 1;
dof = 3;

file_IC = dir('outputs/ex1_*_IC.dat');
if length(file_IC) > 1
    error('Check your outputs, there may be two simulations!');
end
IC  = PetscBinaryRead(fullfile(file_IC(1).folder,file_IC(1).name));
% Get Nx, Ny, and dt from filename
strs = strsplit(file_IC(1).name,'_');
for i = 1 : length(strs)
    if strcmp(strs{i},'Nx')
        Nx = str2num(strs{i+1});
    elseif strcmp(strs{i},'Ny')
        Ny = str2num(strs{i+1});
    elseif strcmp(strs{i},'dt')
        dt = str2num(strs{i+1});
    end
end
x = 1 : Nx;
y = 1 : Ny;
[x,y] = meshgrid(x,y);

% Show Initial Condition
figure;
IC  = reshape(IC,  [dof, length(IC)/dof]);
h0  = IC(1,:);
h0  = reshape(h0,[Nx Ny]);
imagesc(h0); colorbar; caxis([0 1]); colormap(jet);
title('Initial Condition','FontSize',15,'FontWeight','bold');

% Animation of DAM break
imAlpha = ones(size(h0));
imAlpha(1:30,96:105) = 0;
imAlpha(106:200,96:105) = 0;
for i = 1 : Nt
    file  = dir(['outputs/ex1_*_' num2str(i) '.dat']);
    data  = PetscBinaryRead(fullfile(file(1).folder,file(1).name));
    data  = reshape(data,  [dof, length(data)/dof]);
    h     = data(1,:); h     = reshape(h,[Nx Ny]);
    u     = data(2,:); u     = reshape(u,[Nx Ny]);
    v     = data(3,:); v     = reshape(v,[Nx Ny]);
    pause(0.01);
    imagesc(h,'AlphaData',imAlpha); colorbar; caxis([0 10]);
    title(['t = ' num2str(i*dt) 's'],'FontSize',15);
end

h(1:31,95:106)    = NaN;
h(105:200,95:106) = NaN;
u(1:31,95:106)    = 0;
u(105:200,95:106) = 0;
v(1:31,95:106)    = 0;
v(105:200,95:106) = 0;

% Water depth and velocity at last time step
figure; set(gcf,'Position',[10 10 1200 500]);
subplot(1,2,1);
surf(x,y,flipud(h),'LineStyle','none'); hold on;
colormap(jet);
contour3(x,y,flipud(h),20, 'k-', 'LineWidth',1);  
view(30,45);

subplot(1,2,2);
quiver(x,y,flipud(v),flipud(u)); hold on;
fill([95 95 106 106 95],[1 96 96 1 1]        ,[0.5 0.5 0.5],'EdgeColor','none');
fill([95 95 106 106 95],[170 200 200 170 170],[0.5 0.5 0.5],'EdgeColor','none');
xlim([1 200]);ylim([1 200]);
contour(x,y,flipud(h),20,'LineWidth',1);
colorbar; colormap(jet);