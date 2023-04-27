function write_exodus_file_v2(filename, coordx, coordy, coordz, block, sidesets)

% Number of vertices
num_of_nodes = length(coordx);

% Are vertex 2D or 3D?
if ~isempty(coordz)
    num_of_dim = 3;
else
    num_of_dim = 2;
end

% Total number of connections
nblk = length(block);

% Allocate memory
num_el_in_blk = zeros(nblk,1);  % Total number of elements/cells in each connection block
num_nod_per_el = zeros(nblk,1); % Number of nodes/vertices of elements in each block

len_string = 0;

blk_name_present = isfield(block,'name');

for iblk = 1:nblk
    num_el_in_blk(iblk)  = size(block(iblk).connect,1);
    num_nod_per_el(iblk) = size(block(iblk).connect,2);
    
    if (blk_name_present)
        len_string = max([len_string length(block(iblk).name)]);
    end
end

% Total number of elements/cells in the mesh
num_elem = sum(num_el_in_blk);


% Now create the mesh file
ncid_out = netcdf.create(filename,'64BIT_OFFSET');

% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Define dimensions
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
dimid_num_dim      = netcdf.defDim(ncid_out,'num_dim',num_of_dim);
dimid_num_of_nodes = netcdf.defDim(ncid_out,'num_nodes',num_of_nodes);
dimid_num_elem     = netcdf.defDim(ncid_out,'num_elem',num_elem);
dimid_num_elm_blk  = netcdf.defDim(ncid_out,'num_el_blk',nblk);

for iblk = 1:nblk
    dimid_num_el_in_blk(iblk) = netcdf.defDim(ncid_out,sprintf('num_el_in_blk%d',iblk), num_el_in_blk(iblk));
    dimid_num_nod_per_el(iblk) = netcdf.defDim(ncid_out,sprintf('num_nod_per_el%d',iblk),num_nod_per_el(iblk));
end

ss_name_present = false;

if ~isempty(sidesets)

    ss_name_present = isfield(sidesets,'name');
    
    num_side_sets = length(sidesets);
    dimid_num_side_sets = netcdf.defDim(ncid_out,'num_side_sets',num_side_sets);
    
    for iside = 1:num_side_sets
        dimid_num_side_ss(iside) = netcdf.defDim(ncid_out,sprintf('num_side_ss%d',iside),length(sidesets(iside).elem));

        if (ss_name_present)
            len_string = max([len_string length(sidesets(iside).name)]);
        end
    end
else
    num_side_sets = 0;
end

dimid_len_string = netcdf.defDim(ncid_out,'len_string',len_string);


% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Define variables
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
netcdf.defVar(ncid_out,'coordx', 'double', dimid_num_of_nodes);
netcdf.defVar(ncid_out,'coordy', 'double', dimid_num_of_nodes);

if ~isempty(coordz)
    netcdf.defVar(ncid_out,'coordz', 'double', dimid_num_of_nodes);
end

varid = netcdf.defVar(ncid_out,'eb_prop1', 'int', dimid_num_elm_blk);
netcdf.putAtt(ncid_out, varid, 'name', 'ID');

netcdf.defVar(ncid_out,'eb_names','char',[dimid_len_string dimid_num_elm_blk]);
netcdf.defVar(ncid_out,'coor_names','char',[dimid_len_string dimid_num_dim]);

for iblk = 1:nblk
    varid = netcdf.defVar(ncid_out,sprintf('connect%d',iblk), 'int', [dimid_num_nod_per_el(iblk) dimid_num_el_in_blk(iblk)]);
    
    switch num_nod_per_el(iblk)
        case 3
            netcdf.putAtt(ncid_out, varid, 'elem_type', 'TRI3');
        case 4
            netcdf.putAtt(ncid_out, varid, 'elem_type', 'SHELL4');
        otherwise
            error('ERROR: Unsupported element type. Element has max of %d connections',dimid_num_nod_per_el(iblk));
    end
    
end

if (num_side_sets > 0)
    
    varid = netcdf.defVar(ncid_out,'ss_prop1', 'int', dimid_num_side_sets);
    netcdf.putAtt(ncid_out, varid, 'name', 'ID');
    
    for iside = 1:num_side_sets
        netcdf.defVar(ncid_out,sprintf('elem_ss%d',iside), 'int', dimid_num_side_ss(iside));
        netcdf.defVar(ncid_out,sprintf('side_ss%d',iside), 'int', dimid_num_side_ss(iside));
    end
    netcdf.defVar(ncid_out,'ss_names','char',[dimid_len_string dimid_num_side_sets]);
end

varid = netcdf.getConstant('GLOBAL');
[~,user_name]=system('echo $USER');

% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Add global attributes
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
netcdf.putAtt(ncid_out,varid,'api_version',single(4.98));
netcdf.putAtt(ncid_out,varid,'version',single(4.98));
netcdf.putAtt(ncid_out,varid,'floating_point_word_size',int32(8));
netcdf.putAtt(ncid_out,varid,'file_size',int32(1));

netcdf.endDef(ncid_out);
netcdf.close(ncid_out);


% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Write data in the file
% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

ncwrite(filename, 'coordx', coordx);
ncwrite(filename, 'coordy', coordy);
if ~isempty(coordz)
    ncwrite(filename, 'coordz', coordz);
end
ncwrite(filename,'eb_prop1',[1:nblk]);


for iblk = 1:nblk
    ncwrite(filename, sprintf('connect%d',iblk), block(iblk).connect');
end

if (num_side_sets > 0)
    ncwrite(filename, 'ss_prop1', [1:length(sidesets)]);

    for iside = 1:num_side_sets
        ncwrite(filename,sprintf('elem_ss%d',iside), sidesets(iside).elem);
        ncwrite(filename,sprintf('side_ss%d',iside), sidesets(iside).side);
    end
end

if (blk_name_present)
    eb_names = ncread(filename,'eb_names');
    for iblk = 1:nblk
        eb_names(1:length(block(iblk).name),iblk) = block(iblk).name;
    end
    ncwrite(filename,'eb_names',eb_names);
end

if (ss_name_present)
    ss_names = ncread(filename,'ss_names');
    for iside = 1:num_side_sets
        ss_names(1:length(sidesets(iside).name),iside) = sidesets(iside).name;
    end
    ncwrite(filename,'ss_names',ss_names);
end

coor_names = ncread(filename,'coor_names');
coor_names(1,1) = 'x';
coor_names(1,2) = 'y';
if ~isempty(coordz)
    coor_names(1,3) = 'z';
end
ncwrite(filename,'coor_names',coor_names);
end
