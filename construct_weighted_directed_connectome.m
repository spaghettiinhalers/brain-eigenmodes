%% construct_weighted_directed_connectome.m
% construct a weighted directed connectome
% binary connections come from sparse high-res connectomes (mansour et al) + mesh-based gm connections between adjacent regions (pang et al)
% currently only for left hemisphere and cortex, and will only create subject-specific connectomes from subject-specific rs data. this is because hpc is hard

addpath(genpath('matlab_functions'));

% surface
surface_interest = 'fsLR_32k';
hemisphere = 'lh';
mesh_interest = 'midthickness';

% threshold
threshold = 0.01;

[vertices, faces] = read_vtk(sprintf('data/template_surfaces_volumes/%s_%s-%s.vtk', surface_interest, mesh_interest, hemisphere));
surface_to_analyze.vertices = vertices';
surface_to_analyze.faces = faces';

% cortex mask
cortex = dlmread(sprintf('data/template_surfaces_volumes/%s_cortex-%s_mask.txt', surface_interest, hemisphere));
cortex_ind = find(cortex);

num_vertices = length(cortex);
num_modes = 200;

connectome = load(sprintf('data/empirical/S255_high-resolution_group_average_connectome_cortex_nomedial-%s.mat', hemisphere));
connectome = connectome.avgSC_L; % left hemisphere

disp('files loaded.');



%% CONNECTOME + SHORT-RANGE CONNECTIONS
disp('GENERATING UNDIRECTED WEIGHTED CONNECTOME FROM EMPIRICAL CONNECTIVITY');

% short-range connections in grey matter
surface_connectivity = calc_surface_connectivity(surface_to_analyze);
% remove vertices corresponding to medial wall
surface_connectivity = surface_connectivity(cortex_ind, cortex_ind);

rng(1)
imin = 0;
imax = 1e4;
triu_ind = calc_triu_ind(connectome);
triu_val = randi([imin,imax], size(triu_ind,1), size(triu_ind,2));

connectome(triu_ind) = triu_val;
% symmetrize
clear triu_ind triu_val verices faces
connectome = connectome + connectome';
% threshold
connectome_threshold = threshold_edges_proportional(connectome, threshold);

surface_with_connectome = surface_connectivity + (connectome_threshold>0);
surface_with_connectome(surface_with_connectome>0) = 1;

connectome = surface_with_connectome;

% connection lengths of vertices without medial wall
euclidean_dist = squareform(pdist(surface_to_analyze.vertices(cortex_ind,:)));

% estimate true length
sigmoid = @(p, x) p(1) * x + p(2) + p(3) ./ (1 + exp(-p(4) * (x - p(5))));
p = [1, 5, 45, 0.15, 35];
surface_dist = sigmoid(p, euclidean_dist);

disp ('Connection lengths calculated.');

clear cortex
clear connectome_threshold surface_with_connectome euclidean_dist vertices faces surface_to_analyze
clear surface_connectivity

% probability function for connection
Pspace_func = @(scale, distance) exp(-scale*distance);
alpha_empirical = 0.1668; % empirical scale exponent parameter based on structural connectivity

scale = alpha_empirical;
Pspace = Pspace_func(scale, surface_dist);
Pspace(1:1+size(Pspace,1):end) = 0;
Pspace = Pspace/max(Pspace(:));

connectome = connectome .* Pspace;

clear vertices faces surface_to_analyze surface_connectivity
clear triu_ind triu_val connectome_threshold surface_with_connectome
clear euclidean_dist surface_dist sigmoid p Pspace_func Pspace
clear imin imax alpha_empirical scale num_modes mesh_interest threshold

%% ESTIMATE PARCEL-LEVEL DIRECTIONALITY WITH rDCM

addpath(genpath('rdcm'));

n_cortex = size(cortex_ind, 1);
n_parcels = 180;

parc = load('data/parcellations/fsLR_32k_Glasser360-lh.txt');
parc = parc(cortex_ind);
parc = parc - 180; % only if hemisphere = lh

% subject-specific rs data
rfmri_data = ft_read_cifti('data/100206/rfMRI_REST1_LR_Atlas_MSMAll_hp2000_clean_rclean_tclean.dtseries.nii');
rfmri_data = rfmri_data.dtseries(rfmri_data.brainstructure == 1,:);
rfmri_data = rfmri_data(cortex_ind, :);

T = size(rfmri_data, 2);
Y = zeros(n_parcels, T);

for p = 1:n_parcels
    idx = (parc==p);
    Y(p, :) = mean(rfmri_data(idx, :), 1);
end

clear rfmri_data idx p

Y = detrend(Y.', 1);
Y_mean = mean(Y, 1);
Y = Y - Y_mean;
clear Y_mean

% parcelxparcel matrix structure to contain rdcm
% parcel pair has an existing connection if there is at least one vertex-level edge between them
P = sparse((1:n_cortex).', parc, 1, n_cortex, n_parcels);
SC = spones(P' * spones(connectome) * P);
SC = spones(SC + SC');
SC = spones(SC + speye(n_parcels));
clear P

%% estimate parcel-level ec with rs rDCM
rdcm_data = struct();
rdcm_data.y = Y;
rdcm_data.dt = 0.72;
rdcm_data.name = arrayfun(@(p) sprintf('Glasser_%03d', p), 1:n_parcels,'UniformOutput',false); % name parcels

rdcm_model = struct();
rdcm_model.a = full(SC);
DCM = tapas_rdcm_model_specification(rdcm_data, [], rdcm_model);
DCM.M.noprint = 1;

rdcm_options = struct();
rdcm_options.filter_str = 1;
rdcm_options.visualize = 0;
rdcm_options.compute_signal = 0;
rdcm_options.evalCp = 0;

[rdcm_output, ~] = tapas_rdcm_estimate(DCM, 'r', rdcm_options, 1);

EC = double(rdcm_output.Ep.A');
% EC(EC < 0) = 0;
EC = abs(EC);
EC(1:n_parcels+1:end) = 0;

clear Y SC rdcm_data rdcm_model DCM rdcm_options rdcm_output

%% ec -> parcel-level directional asymmetry

EC_sum = EC + EC';
asymmetry = nan(n_parcels,n_parcels);
resolved_pair = EC_sum > 0;
asymmetry(resolved_pair) = EC(resolved_pair) ./ EC_sum(resolved_pair);
asymmetry(1:n_parcels+1:end) = 0.5;

clear EC EC_sum resolved_pair

%% project parcel asymmetry back to weighted vertex connectome
[vertex_i,vertex_j,undir_wt] = find(triu(connectome,1));
parc_i = parc(vertex_i);
parc_j = parc(vertex_j);
num_edges = numel(undir_wt);
forward_wt = zeros(num_edges, 1);
reverse_wt = zeros(num_edges, 1);

same_parc = parc_i==parc_j;
forward_wt(same_parc) = 0.5 .* undir_wt(same_parc);
reverse_wt(same_parc) = 0.5 .* undir_wt(same_parc);

interparc_index = find(~same_parc);
edge_asymmetry = asymmetry(sub2ind([n_parcels n_parcels], parc_i(interparc_index), parc_j(interparc_index)));
resolved_edge = isfinite(edge_asymmetry);

unresolved_idx = interparc_index(~resolved_edge);
forward_wt(unresolved_idx) = 0.5 .* undir_wt(unresolved_idx);
reverse_wt(unresolved_idx) = 0.5 .* undir_wt(unresolved_idx);

direction_floor = eps(1);
resolved_asymmetry = edge_asymmetry(resolved_edge);
resolved_asymmetry = min(max(resolved_asymmetry, direction_floor), 1 - direction_floor);
clear unresolved_idx direction_floor
clear asymmetry parc_i parc_j parc num_edges
clear connectome cortex_ind

resolved_idx = interparc_index(resolved_edge);
clear same_parc interparc_index edge_asymmetry resolved_edge

forward_wt(resolved_idx) = undir_wt(resolved_idx) .* resolved_asymmetry;
reverse_wt(resolved_idx) = undir_wt(resolved_idx) .* (1 - resolved_asymmetry);

clear resolved_idx
clear resolved_asymmetry

%% add medial wall

dir_connectome = sparse(vertex_i, vertex_j, forward_wt, n_cortex, n_cortex);
clearvars -except dir_connectome vertex_i vertex_j reverse_wt n_cortex

dir_connectome = dir_connectome + sparse(vertex_j, vertex_i, reverse_wt, n_cortex, n_cortex);
clearvars -except dir_connectome

dir_connectome = full(dir_connectome);

% subject-specific
save('data/100206/dir_connectome.mat', 'dir_connectome', '-v7.3');