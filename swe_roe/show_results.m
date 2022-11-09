function show_results(example_number)
% Plots output of SWE from examples in this directory
%
% Examples
%   show_results(1)

close all

if ~exist('PetscBinaryRead')
    error(['Please add PETSc MATLAB files before calling this script via: ' char(10) ...
        'addpath <path-to-petsc>/share/petsc/matlab/'])
end

switch example_number
    case {1,2}
    otherwise
        error('Invalid example_number. It can only be 1 or 2');
end

Lx = 200; % [m]
Ly = 200; % [m]
Nt  = length(dir(sprintf('outputs/ex%d_Nx*.dat',example_number))) - 1;
dof = 3;

file_IC = dir(sprintf('outputs/ex%d_*_IC.dat',example_number));
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

dx = Lx / Nx;
dy = Ly / Ny;
x = dx : dx : Lx;
y = dy : dy : Ly;
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
imAlpha(1:30/dx,95/dy+1:105/dy) = 0;
imAlpha(105/dy+1:200/dy,95/dy+1:105/dy) = 0;
for i = 0 : Nt-1
    file  = dir(['outputs/ex' num2str(example_number) '_*_' num2str(i) '.dat']);
    data  = PetscBinaryRead(fullfile(file(1).folder,file(1).name));
    data  = reshape(data,  [dof, length(data)/dof]);
    h     = data(1,:); h     = reshape(h,[Nx Ny]);
    u     = data(2,:); u     = reshape(u,[Nx Ny]);
    v     = data(3,:); v     = reshape(v,[Nx Ny]);
    pause(0.01);
    imagesc(h,'AlphaData',imAlpha); colorbar; caxis([0 10]);
    title([ 'ex' num2str(example_number) ': t = ' num2str(i*dt) 's'],'FontSize',15);
end

h(1:30/dx+1,95/dy:105/dy+1)     = NaN;
h(105/dx:200/dy,95/dy:105/dy+1) = NaN;
u(1:30/dx+1,95/dy:105/dy+1)     = 0;
u(105/dx:200/dx,95/dy:105/dy+1) = 0;
v(1:30/dx+1,95/dy:105/dy+1)     = 0;
v(105/dx:200/dx,95/dy:105/dy+1) = 0;

% Water depth and velocity at last time step
figure; set(gcf,'Position',[10 10 1200 500]);
subplot(1,2,1);
surf(x,y,flipud(h),'LineStyle','none'); hold on;
colormap(jet);
contour3(x,y,flipud(h),20, 'k-', 'LineWidth',1);  
view(30,45);
title(['ex' num2str(example_number)])

subplot(1,2,2);
quiver(x,y,flipud(v),flipud(u)); hold on;
fill([95 95 106 106 95],[1 96 96 1 1]        ,[0.5 0.5 0.5],'EdgeColor','none');
fill([95 95 106 106 95],[170 200 200 170 170],[0.5 0.5 0.5],'EdgeColor','none');
xlim([1 200]);ylim([1 200]);
contour(x,y,flipud(h),20,'LineWidth',1);
colorbar; colormap(jet);
title(['ex' num2str(example_number)])
