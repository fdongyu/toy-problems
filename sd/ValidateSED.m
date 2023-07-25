clear;close all;clc;

% The following script is to validate RDycore againt tRIBS-FEaST with
% Proffitt rainfall-induced erosion example. 
%
% Run ex2c.c with following command line: 
% mpiexec -n 1 ./ex2c -savef true -output_prefix proffitt \
%                     -mesh ../share/meshes/proffitt.exo  \
%                     -dt 0.05 -Nt 60000 -sed true -mannings_n 0.06 \
%                     -ic 2 -h_ic 0.01

addpath('/Users/xudo627/developments/petsc/share/petsc/matlab/');

Rx   = ncread('../share/meshes/proffitt.exo','coordx');
Ry   = ncread('../share/meshes/proffitt.exo','coordy');
Rz   = ncread('../share/meshes/proffitt.exo','coordz');
Rtri = ncread('../share/meshes/proffitt.exo','connect1');
Rxv  = Rx(Rtri);
Ryv  = Ry(Rtri);

% Read outputs from RDycore sediment simulations
Rarray  = PetscBinaryRead('outputs/proffitt_dt_0.050000_final_solution.dat');
Rarray  = reshape(Rarray,[13 size(Rxv,2)]);
rdycore_Sout = dlmread('./outputs/proffitt_rank0.Soutlet');
rdycore_Qout = dlmread('./outputs/proffitt_rank0.Qoutlet');

% Read outputs from tRIBS-FEaST simulations
Tarray  = dlmread('Sediment_Proffitt_10mm_calib8.Spatial2');
node   = dlmread('Sediment_Proffitt.nodes',' ',2,0);
Ttri   = dlmread('Sediment_Proffitt.tri',' ',2,0);
voleta = dlmread('Sediment_Proffitt_10mm_calib8.Voleta');
Tx     = node(:,1); Ty = node(:,2);
Txv    = Tx(Ttri(:,1:3)+1)'; 
Tyv    = Ty(Ttri(:,1:3)+1)'; 
tribs_Sout   = dlmread('Sediment_Proffitt_10mm_calib8.QoutletS');
tribs_Qout   = dlmread('Sediment_Proffitt_10mm_calib8.Qoutlet');

figure;
ind = 50;
ht  = voleta((ind-1)*1170+1:ind*1170,5);
ut  = voleta((ind-1)*1170+1:ind*1170,3);
vt  = voleta((ind-1)*1170+1:ind*1170,4);
subplot(2,3,1);
patch(Txv,Tyv,ht,'linestyle','none'); colorbar; 
title(gca,'TRIBS H','FontSize',15,'FontWeight','bold');
clim([1e-3 7e-3]);
subplot(2,3,2);
patch(Txv,Tyv,ut,'linestyle','none'); colorbar; 
title(gca,'TRIBS U','FontSize',15,'FontWeight','bold');
clim([0 0.03]);
subplot(2,3,3);
patch(Txv,Tyv,vt,'linestyle','none'); colorbar; 
title(gca,'TRIBS V','FontSize',15,'FontWeight','bold');

subplot(2,3,4);
patch(Rxv,Ryv,Rarray(1,:),'linestyle','none'); colorbar; 
title(gca,'RDycore H','FontSize',15,'FontWeight','bold');
clim([1e-3 7e-3]);
subplot(2,3,5);
patch(Rxv,Ryv,Rarray(2,:)./Rarray(1,:),'linestyle','none'); colorbar; 
title(gca,'RDycore U','FontSize',15,'FontWeight','bold');
clim([0 0.03]);
subplot(2,3,6);
patch(Rxv,Ryv,Rarray(3,:)./Rarray(1,:),'linestyle','none'); colorbar; 
title(gca,'RDycore V','FontSize',15,'FontWeight','bold');


figure; set(gcf,'Position',[10 10 600 1200]);
for i = 1 : 10
    subplot(10,2,(i-1)*2+1)
    patch(Txv,Tyv,Tarray((ind-1)*1170+1:ind*1170,i+1),'linestyle','none'); colorbar;
    %a = clim;
    subplot(10,2,(i-1)*2+2)
    patch(Rxv,Ryv,Rarray(3+i,:)./Rarray(1,:),'linestyle','none'); colorbar;
    %clim([a(1) a(2)]);
end

figure; set(gcf,'Position',[10 10 800 400]);
subplot(1,2,1);
plot(tribs_Qout(:,end),'k-','LineWidth',2); hold on;
plot(rdycore_Qout(2:end,2),'ro'); grid on;
xlabel('[min]','FontSize',15,'FontWeight','bold');
add_title(gca,'Flow discharge [m^{3}/s]');
legend('tRIBS','RDycore','FontSize',14,'FontWeight','bold');
subplot(1,2,2);
plot(tribs_Sout(:,end),'k-','LineWidth',2); hold on;
plot(rdycore_Sout(2:end,12),'ro'); grid on;
xlabel('[min]','FontSize',15,'FontWeight','bold');
add_title(gca,'Sediment discharge [kg/s]');

figure;set(gcf,'Position',[10 10 1200 600]);
for i = 1 : 10
    subplot(2,5,i);
    plot(tribs_Sout(:,6+i),'k-','LineWidth',2); hold on;
    plot(rdycore_Sout(2:end,i+1),'ro'); grid on
    title(['Sediment class ' num2str(i)]);
    if i == 1
        legend('tRIBS','RDycore','FontSize',14,'FontWeight','bold');
    end
    xlabel('[min]','FontSize',15,'FontWeight','bold');
end
