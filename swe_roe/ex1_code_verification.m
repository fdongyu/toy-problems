clear;close all;clc

% Run several simulations before runing the following scripts
% mpiexec -n 40 ./ex1 -hd 5.0 -save 2 -b -dx 10 -dy 10 -dt 0.005
% mpiexec -n 40 ./ex1 -hd 5.0 -save 2 -b -dx 2 -dy 2 -dt 0.005
% mpiexec -n 40 ./ex1 -hd 5.0 -save 2 -b -dx 1 -dy 1 -dt 0.005
% mpiexec -n 40 ./ex1 -hd 5.0 -save 2 -b -dx 0.5 -dy 0.5 -dt 0.005
% mpiexec -n 40 ./ex1 -hd 5.0 -save 2 -b -dx 0.2 -dy 0.2 -dt 0.005
% mpiexec -n 40 ./ex1 -hd 5.0 -save 2 -b -dx 0.1 -dy 0.1 -dt 0.005 <-- assuming this is exact solution

% User need to provide petsc directory
addpath('/Users/xudo627/developments/petsc/share/petsc/matlab/');
Lx = 200; % [m]
Ly = 200; % [m]
dof = 3;

filename = 'outputs/ex1_Nx_2000_Ny_2000_dt_0.005000_1440.dat';
data = PetscBinaryRead(filename);
strs = strsplit(filename,'_');
for i = 1 : length(strs)
    if strcmp(strs{i},'Nx')
        Nx = str2num(strs{i+1});
    elseif strcmp(strs{i},'Ny')
        Ny = str2num(strs{i+1});
    elseif strcmp(strs{i},'dt')
        dt = str2num(strs{i+1});
    end
end

figure;
data   = reshape(data,  [dof, length(data)/dof]);
hexact = reshape(data(1,:),[Nx Ny]);
uexact = reshape(data(1,:),[Nx Ny]);
vexact = reshape(data(1,:),[Nx Ny]);

imagesc(hexact); colorbar; caxis([0 10]); colormap(jet);

Nxexact = 2000;

dx     = [];
err_L1  = NaN(5,3);
err_L2  = NaN(5,3);
err_Max = NaN(5,3);

k = 1;
for Nx = [20 100 200 400 1000]
    disp(['Nx = ' num2str(Nx)]);
    Ny = Nx;
    dx = [dx; Lx/Nx];
    filename = ['outputs/ex1_Nx_' num2str(Nx) '_Ny_' num2str(Ny) '_dt_0.005000_1440.dat'];
    data = PetscBinaryRead(filename);
    data = reshape(data,  [dof, length(data)/dof]);
    h    = reshape(data(1,:),[Nx Ny]);
    u    = reshape(data(1,:),[Nx Ny]);
    v    = reshape(data(1,:),[Nx Ny]);
    scale_factor = Nxexact / Nx;
    hexact_tmp = convert_res(hexact,1,scale_factor) ./ scale_factor^2;
    uexact_tmp = convert_res(uexact,1,scale_factor) ./ scale_factor^2;
    vexact_tmp = convert_res(vexact,1,scale_factor) ./ scale_factor^2;
    
    for i = 1 : 3
        if i == 1
            err = abs(h - hexact_tmp);
        elseif i == 2
            err = abs(u.*h - uexact_tmp.*hexact_tmp);
        elseif i == 3
            err = abs(v.*h - vexact_tmp.*hexact_tmp);
        end
        err_L1(k,i)  = sum(abs(err(:)))/(Nx*Ny);
        err_L2(k,i)  = sqrt(sum(err(:).^2)/(Nx*Ny));
        err_Max(k,i) = max(abs(err(:)));
    end
    
    k = k + 1;
end

figure; set(gcf,'Position',[10 10 900 300]);
varnames = {'h','uh','vh'};
for i = 1 : 3
    subplot(1,3,i);
    loglog(dx, err_L1(:,i) ,'bo-','LineWidth',2); hold on; grid on;
    loglog(dx, err_L2(:,i) ,'g*-','LineWidth',2); 
    loglog(dx, err_Max(:,i),'rx-','LineWidth',2); 
    ydum = ylim;
    loglog([dx(end-1) dx(1)], ydum(1).*[1, dx(1)/dx(end-1)],'k--','LineWidth',2);
    title(varnames{i},'FontSize',15,'FontWeight','bold');
    if i == 1
        leg = legend('1-norm','2-norm','max-norm','first order');
    end
end


function Z = convert_res(A,u,d)
    %u = 5;       %upsampling factor
    %d = 12;      %downsampling factor
    t = d/u;     %reduction factor
    [m,n]=size(A);
    L1=speye(m); L2=speye(round(m/t))/u;
    R1=speye(n); R2=speye(round(n/t))/u;
    L=repelem(L2,1,d) * repelem(L1,u,1);
    R=(repelem(R2,1,d) * repelem(R1,u,1)).';
    Z= L*(A*R);
end

