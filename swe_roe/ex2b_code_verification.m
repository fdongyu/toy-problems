clear;close all;clc;

addpath('/Users/xudo627/developments/petsc/share/petsc/matlab/');

prefix  = {'dambreak500x1000','dambreak250x500','dambreak100x200', ...
           'dambreak50x100','dambreak5x10'};

meshes  = {'DamBreak_grid500x1000.exo','DamBreak_grid250x500.exo', ...
           'DamBreak_grid100x200.exo','DamBreak_grid50x100.exo',   ...
           'DamBreak_grid5x10.exo'};

dxs = {'dx = 1 [m]','dx = 2 [m]','dx = 5 [m]','dx = 10 [m]','dx = 100 [m]'};

figure(1); set(gcf,'Position',[10 10 400 1000]);

for i = 1 : length(meshes)
    tri    = ncread(['/Users/xudo627/developments/toy-problems/share/meshes/' meshes{i}],'connect1');
    coordx = ncread(['/Users/xudo627/developments/toy-problems/share/meshes/' meshes{i}],'coordx');
    coordy = ncread(['/Users/xudo627/developments/toy-problems/share/meshes/' meshes{i}],'coordy');
    trix   = coordx(tri);
    triy   = coordy(tri);

    data = PetscBinaryRead(['./outputs/ex2b_' prefix{i} '_dt_0.100000_final_solution.dat']);
    data = reshape(data,[3,length(data)/3]);
    h = data(1,:);
    uh= data(2,:);
    vh = data(3,:);

    if i == 1
        h0 = h;
        uh0 = uh;
        vh0 = vh;
        trix0 = trix;
        triy0 = triy;
    end

    hinterp  = griddata(nanmean(trix,1),nanmean(triy,1),h,nanmean(trix0,1),nanmean(triy0,1),'nearest');
    uhinterp = griddata(nanmean(trix,1),nanmean(triy,1),uh,nanmean(trix0,1),nanmean(triy0,1),'nearest');
    vhinterp = griddata(nanmean(trix,1),nanmean(triy,1),vh,nanmean(trix0,1),nanmean(triy0,1),'nearest');
    if i > 1
        err(i-1,1) = norm(hinterp - h0,2);
        err(i-1,2) = norm(uhinterp - uh0,2);
        err(i-1,3) = norm(vhinterp - vh0,2);
    end
    ax(i) = subplot(5,1,i);
    patch(trix,triy,h,'LineStyle','none'); colormap("jet"); clim([5 10]); hold on;
    xlim([0 1000]); ylim([0 500]);
%     fill([400 600 600 400],[0   0   200 200],[0.5 0.5 0.5],'EdgeColor','none');
%     fill([400 600 600 400],[400 400 500 500],[0.5 0.5 0.5],'EdgeColor','none');
    title(dxs{i},'FontSize',15,'FontWeight','bold');
    if i == 5
        cb = colorbar('east');
    else
        set(gca,'xtick',[],'ytick',[]);
    end
end
for i = 1 : 5
    ax(i).Position(1) = ax(i).Position(1) - 0.05;
end

x0 = ax(5).Position(1) + ax(5).Position(3) + 0.02;
y0 = ax(5).Position(2);
w0 = 0.03;
d0 = ax(1).Position(2) + ax(1).Position(4) - ax(5).Position(2);
cb.Position = [x0 y0 w0 d0];
cb.FontSize =13;
title(cb,'h [m]','FontSize',18,'FontWeight','bold');

figure(2); set(gcf,'Position',[10 10 400 1000]);
h0(nanmean(trix0)<= 400) = 10;
h0(nanmean(trix0) > 400) = 5;
subplot(5,1,1);
patch(trix0,triy0,h0,'LineStyle','none'); colormap("jet"); clim([5 10]); hold on;
xlim([0 1000]); ylim([0 500]);
text(150,240,'10 [m]','Color','w','FontSize',15,'FontWeight','bold');
text(730,240,'5 [m]','Color','w','FontSize',15,'FontWeight','bold');

figure; set(gcf,'Position',[10 10 1200 400]);
labs = {'h','uh','vh'};
for i = 1 : 2
    subplot(1,2,i);
    if i == 1
        h1(1) = loglog([2; 5; 10; 100],err(:,i),'rx','LineWidth',2); hold on; grid on;
    elseif i == 2
        h2(1) = loglog([2; 5; 10; 100],err(:,2),'bd','LineWidth',2); hold on; grid on;
        h2(2) = loglog([2; 5; 10; 100],err(:,3),'go','LineWidth',2); hold on; grid on;
    end
    
    if i == 1
        a = (err(end,i) - err(1,i)) / (100 -2);
        b = err(1,i) - a*2;
        h1(2) = loglog([1 100],[a+b 100*a + b],'k--','LineWidth',2);
    else
        a = (err(end,i) - err(1,i)) / (100 -2);
        b = err(1,i) - a*2;
        h1 = loglog([1 100],[a+b 100*a + b],'k--','LineWidth',2);

        a = (err(end,3) - err(1,3)) / (100 -2);
        b = err(1,3) - a*2;
        h1 = loglog([1 100],[a+b 100*a + b],'k:','LineWidth',2);
    end
    set(gca,'FontSize',13)
    xlabel('dx [m]','FontSize',18,'FontWeight','bold');
    ylabel('error','FontSize',18,'FontWeight','bold');
    if i == 1
        legend(h1,{'h','1st order'},'FontSize',18,'FontWeight','bold');
    else
        legend(h2,{'uh','vh'},'FontSize',18,'FontWeight','bold');
    end
end