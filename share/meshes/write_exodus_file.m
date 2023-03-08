function write_exodus_file(coordx, coordy, coordz, tri, filename)

num_of_nodes = length(coordx);
if ~isempty(coordz)
    num_of_dim = 3;
else
    num_of_dim = 2;
end

connect1 = tri';

ncid_out = netcdf.create(filename,'64BIT_OFFSET');

% Define dimensions
dimids = zeros(8);
dimids(1) = netcdf.defDim(ncid_out,'num_dim',num_of_dim);
dimids(2) = netcdf.defDim(ncid_out,'num_nodes',num_of_nodes);
dimids(3) = netcdf.defDim(ncid_out,'num_elem',size(connect1,2));
dimids(4) = netcdf.defDim(ncid_out,'num_el_blk',1);
dimids(5) = netcdf.defDim(ncid_out,'num_el_in_blk1',size(connect1,2));
dimids(6) = netcdf.defDim(ncid_out,'num_nod_per_el1',size(connect1,1));
dimids(7) = netcdf.defDim(ncid_out,'time_step',netcdf.getConstant('NC_UNLIMITED'));
dimids(8) = netcdf.defDim(ncid_out,'num_elem_var',11);

% Define variable
varid(1) = netcdf.defVar(ncid_out,'eb_prop1', 'int', dimids(4));
varid(2) = netcdf.defVar(ncid_out,'coordx', 'double', dimids(2));
varid(3) = netcdf.defVar(ncid_out,'coordy', 'double', dimids(2));
if ~isempty(coordz)
    varid(4) = netcdf.defVar(ncid_out,'coordz', 'double', dimids(2));
    varid(5) = netcdf.defVar(ncid_out,'connect1', 'int', [dimids(6) dimids(5)]);
    connect_id = varid(5);
else
    varid(4) = netcdf.defVar(ncid_out,'connect1', 'int', [dimids(6) dimids(5)]);
    connect_id = varid(4);
end
netcdf.putAtt(ncid_out, varid(1), 'name', 'ID');

switch size(connect1,1)
    case 3
        netcdf.putAtt(ncid_out, connect_id, 'elem_type', 'TRI');
    case 4
        netcdf.putAtt(ncid_out, connect_id, 'elem_type', 'QUAD');
    case 6
        netcdf.putAtt(ncid_out, connect_id, 'elem_type', 'WEDGE');
    case 8
        netcdf.putAtt(ncid_out, connect_id, 'elem_type', 'HEX');
    otherwise
        error(sprintf('ERROR: Unsupported element type. Element has maximum of %d connections', size(connect1,1)));
end

varid = netcdf.getConstant('GLOBAL');
[~,user_name]=system('echo $USER');

% Add global attributes
netcdf.putAtt(ncid_out,varid,'api_version',4.93);
netcdf.putAtt(ncid_out,varid,'version',4.93);
netcdf.putAtt(ncid_out,varid,'floating_point_word_size',8);
netcdf.putAtt(ncid_out,varid,'file_size',1);

netcdf.endDef(ncid_out);

netcdf.close(ncid_out);

% Write data into the file
ncwrite(filename, 'eb_prop1', 1);
ncwrite(filename, 'coordx', coordx);
ncwrite(filename, 'coordy', coordy);
if ~isempty(coordz)
    ncwrite(filename, 'coordz', coordz);
end

ncwrite(filename, 'connect1', connect1);

end
