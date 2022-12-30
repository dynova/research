% Runtime: ~3 minutes

%% Preliminaries
e_vec = 6:12; % vector of exponents 'k' in 2^k
len = length(e_vec);
M_vec = 2.^e_vec; % vector of 'M' values
h_vec = 1 ./ M_vec;
num_tests = 3; % Number of test cases
num_methods = 2; % Number of finite difference methods
num_norms = 3; % Number of norm cases
timesteps_vec = M_vec.*30;

%% Preallocations with cell arrays
[c_grid, data_e, data_c, data_i, data_i1, data_i2] = deal(cell(1, len));
f = {@Psi_1, @Psi_2, @Psi_3};

for ii = 1:len
    [data_e{ii}, data_c{ii}, data_i{ii}, data_i1{ii}, data_i2{ii}] =...
        deal(cell(num_tests, num_methods));
    for jj = 1:num_tests*num_methods
        [data_e{ii}{jj}, data_c{ii}{jj}, data_i{ii}{jj}, data_i1{ii}{jj},...
            data_i2{ii}{jj}] = deal(zeros(1, M_vec(ii)));
    end
end

%% Finite difference schemes
for ind = 1:len
    M = M_vec(ind);
    h = h_vec(ind);
    c_grid{ind} = h*(1:M)-h/2;
    timesteps = timesteps_vec(ind);
    coeff = [-9/400, 369/400, 49/400, -9/400];
    iim2 = mod(2:M+1, M) + 1;
    iim1 = mod(1:M, M) + 1;
    ii = 1:M;
    iip1 = mod(-1:M-2, M) + 1;
    idxs = [iim2; iim1; ii; iip1];
    for kk = 1:num_tests
        for jj = 1:num_methods
            data_e{ind}{kk,jj} = f{kk}(c_grid{ind});
            data_c{ind}{kk,jj} = f{kk}(c_grid{ind});
            data_i{ind}{kk,jj} = f{kk}(c_grid{ind});
            data_i1{ind}{kk,jj} = f{kk}(c_grid{ind});
            data_i2{ind}{kk,jj} = f{kk}(c_grid{ind});
        end
        for t = 1:timesteps
            data_c{ind}{kk,1} = coeff*data_i{ind}{kk,1}(idxs);
            data_i{ind}{kk,1} = data_c{ind}{kk,1};
            sgn = sign(data_i1{ind}{kk,2}(iim1) - data_i1{ind}{kk,2}(iip1));
            phi = (data_i1{ind}{kk,2}(iim1) - data_i1{ind}{kk,2}) .* (data_i1{ind}{kk,2} - data_i1{ind}{kk,2}(iip1));
            theta = min([2 * abs(data_i1{ind}{kk,2} - data_i1{ind}{kk,2}(iip1)); ...
                0.5 * abs(data_i1{ind}{kk,2}(iim1) - data_i1{ind}{kk,2}(iip1)); ...
                2 * abs(data_i1{ind}{kk,2}(iim1) - data_i1{ind}{kk,2})]);
            delta = (phi > 0) .* sgn .* theta;
            data_i2{ind}{kk,2} = data_i1{ind}{kk,2} + (1/20) * delta;
            data_c{ind}{kk,2} = data_i1{ind}{kk,2} + (9/10) * (data_i2{ind}{kk,2}(iip1) - data_i2{ind}{kk,2});
            data_i1{ind}{kk,2} = data_c{ind}{kk,2};
        end
    end
end

%% Error computations
format long
[L1, L2, Linf] = deal(zeros(len,num_tests,num_methods));

for ii = 1:len
    for jj = 1:num_tests
        for kk = 1:num_methods
            L1(ii,jj,kk) = Compute_One_Norm(data_e{ii}{jj,kk},data_c{ii}{jj,kk},M_vec(ii));
            L2(ii,jj,kk) = Compute_Two_Norm(data_e{ii}{jj,kk},data_c{ii}{jj,kk},M_vec(ii));
            Linf(ii,jj,kk) = Compute_Max_Norm(data_e{ii}{jj,kk},data_c{ii}{jj,kk});
        end
    end
end

%% Rate computations
N = {'L1','L2','Linf'};
rates = zeros(len-1,num_tests,num_methods,num_norms);
for hh = 1:len-1
    for ii = 1:num_tests
        for jj = 1:num_methods
            for kk = 1:num_norms
                rates(hh,ii,jj,kk) = log2(eval([eval('N{kk}') '(hh,ii,jj)'])/eval([eval('N{kk}') '(hh+1,ii,jj)']));
            end
        end
    end
end

%% Table generation
MM = cell(num_norms,num_tests);
fid = zeros(num_norms,num_tests);
T = {'GAUSSIAN_', 'SEMICIRCLE_', 'SQUARE_WAVE_'};

for ii = 1:num_norms
    for jj = 1:num_tests
        MM{ii,jj} = zeros(len,4);
        MM{ii,jj}(:,1) = e_vec;
        MM{ii,jj}(:,2) = h_vec;
        MM{ii,jj}(:,3) = eval([eval('N{ii}') '(:,jj,1)']);
        MM{ii,jj}(:,4) = eval([eval('N{ii}') '(:,jj,2)']);
        fid(ii,jj) = fopen(['amallela_' T{jj} N{ii} '_ERROR.csv'],'wt');
        fprintf(fid(ii,jj),'%d, %.6e, %.6e, %.6e',MM{ii,jj}(1,:));
        for ll = 2:len
            fprintf(fid(ii,jj),'\n%d, %.6e, %.6e, %.6e',MM{ii,jj}(ll,:));
        end
        fclose(fid(ii,jj));
    end
end

%% Generation of state data
X = {'FROMM_', 'FvL_'};
R = '4096';
fid = zeros(num_methods,num_tests);
for ii = 1:num_methods
    for kk = 1:num_tests
        fid(ii,kk) = fopen(['amallela_' T{kk} X{ii} R '_STATE.txt'],'wt');
        for ll = 1:str2double(R)
            fprintf(fid(ii,kk),'%.15e,%.15e \n',[c_grid{end}(ll) data_c{end}{kk,ii}(ll)]);
        end
        fclose(fid(ii,kk));
    end
end

% %% Output of plots for LaTeX report
% init_conds = {'Gaussian Pulse:', 'Semicircle:', 'Square Wave:'};
% methods = {'FROMM', 'FvL'};
% colors = {'k','b','r', 'g'};
% num_resolutions = 4;
% counter = 1;
% for jj = 1:num_methods
%     for ii = 1:num_tests
%         figure(counter)
%         clf;
%         counter = counter + 1;
%         for kk = 1:num_resolutions
%             hold on
%             plot(c_grid{2*kk},data_e{2*kk}{ii,jj},colors{kk},'linewidth',2)
%             plot(c_grid{2*kk},data_c{2*kk}{ii,jj},colors{kk},'linewidth',2)
%             title(strcat(init_conds{ii},{' '},methods{jj}),'fontsize',16);
%             legend('Exact: M = 2^{8}', 'Computed: M = 2^{8}', 'Exact: M = 2^{10}', 'Computed: M = 2^{10}',...
%                 'Exact: M = 2^{12}', 'Computed: M = 2^{12}', 'Exact: M = 2^{14}', 'Computed: M = 2^{14}' ,...
%                 'Location','Best');
%             hold off
%         end
%         saveas(gcf,strcat('Figure_', num2str(counter-1),'.pdf'));
%     end
% end

%% Subroutines
function psi_1 = Psi_1(z)
psi_1 = exp(-256.*(z-0.5).^2);
end

function psi_2 = Psi_2(z)
psi_2 = (0.25-(z-0.5).^2).^0.5;
end

function psi_3 = Psi_3(z)
psi_3 = and(z>=0.25,z<=0.75);
end

function one_norm = Compute_One_Norm(u, v, M)
one_norm = sum(abs(u-v))/M;
end

function two_norm = Compute_Two_Norm(u, v, M)
two_norm = sqrt(sum(abs(u-v).^2)/M);
end

function max_norm = Compute_Max_Norm(u, v)
max_norm = max(abs(u - v));
end