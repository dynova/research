% Runtime: ~100 minutes

%% Preliminaries
e_vec = 6:14; % vector of exponents 'k' in 2^k
len = length(e_vec);
M_vec = 2.^e_vec; % vector of 'M' values
h_vec = 1 ./ M_vec;
num_tests = 3; % Number of test cases
num_methods = 2; % Number of finite difference methods
num_norms = 3; % Number of norm cases
FCT_timesteps_vec = M_vec.*30;

%% Preallocations with cell arrays
[c_grid, data_e, data_c, data_i] = deal(cell(1, len));
f = {@Psi_1, @Psi_2, @Psi_3};

for ii = 1:len
    [data_e{ii}, data_c{ii}, data_i{ii}] = deal(cell(num_tests, num_methods));
    for jj = 1:num_tests*num_methods
        [data_e{ii}{jj}, data_c{ii}{jj}, data_i{ii}{jj}] = deal(zeros(1, M_vec(ii)));
    end
end

%% Finite difference schemes
for ind = 1:len
    M = M_vec(ind);
    h = h_vec(ind);
    c_grid{ind} = h*(1:M)-h/2;
    FCT_timesteps = FCT_timesteps_vec(ind);
    coeff = [-937/30000, 529/2500, 4363/5000, -413/7500, 21/10000];
    iim2 = mod(2:M+1, M) + 1;
    iim1 = mod(1:M, M) + 1;
    ii = 1:M;
    iip1 = mod(-1:M-2, M) + 1;
    iip2 = mod(-2:M-3, M) + 1;
    idxs = [iim2; iim1; ii; iip1; iip2];
    for kk = 1:num_tests
        for jj = 1:num_methods
            data_e{ind}{kk,jj} = f{kk}(c_grid{ind});
            data_c{ind}{kk,jj} = f{kk}(c_grid{ind});
            data_i{ind}{kk,jj} = f{kk}(c_grid{ind});
        end
        for t1 = 1:FCT_timesteps
            sgn = sign((3/20) * (data_i{ind}{kk,1}(iip1) - data_i{ind}{kk,1}));
            theta = min([sgn .* (10/3) .* ((1/10) * data_i{ind}{kk,1}(iip2) + (4/5) * ...
                data_i{ind}{kk,1}(iip1) - (9/10) * data_i{ind}{kk,1}(ii)); ...
                sgn .* (10/3) .* ((1/10) * data_i{ind}{kk,1}(ii) + (4/5) * ...
                data_i{ind}{kk,1}(iim1) - (9/10) * data_i{ind}{kk,1}(iim2)); ...
                (3/20) * abs(data_i{ind}{kk,1}(iip1) - data_i{ind}{kk,1}(ii))]);
            c_flux = (theta > 0) .* sgn .* theta;
            data_c{ind}{kk,1} = data_i{ind}{kk,1} + (9/10) * ...
                (data_i{ind}{kk,1}(iim1)-data_i{ind}{kk,1}(ii)) + ...
                (3/10) * (c_flux(iim1) - c_flux);
            data_i{ind}{kk,1} = data_c{ind}{kk,1};
        end
        for t2 = 1:FCT_timesteps*4.5
            data_c{ind}{kk,2} = coeff*data_i{ind}{kk,2}(idxs);
            data_i{ind}{kk,2} = data_c{ind}{kk,2};
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

for ind = 1:len
    for jj = 1:num_methods*num_tests
        data_c{ind}{jj} = fliplr(data_c{ind}{jj});
    end
end

%% Generation of state data
X = {'FCT_', 'LW4_'};
R = '4096';
fid = zeros(num_methods,num_tests);
for ii = 1:num_methods
    for kk = 1:num_tests
        fid(ii,kk) = fopen(['amallela_' T{kk} X{ii} R '_STATE.txt'],'wt');
        for ll = 1:str2double(R)
            fprintf(fid(ii,kk),'%.15e,%.15e \n',[c_grid{end-2}(ll) data_c{end-2}{kk,ii}(ll)]);
        end
        fclose(fid(ii,kk));
    end
end

%% Output of plots for LaTeX report
init_conds = {'Gaussian Pulse:', 'Semicircle:', 'Square Wave:'};
methods = {'FCT', 'LW4'};
colors = {'y','g','b','r'};
num_resolutions = 4;
counter = 1;
for jj = 1:num_methods
    for ii = 1:num_tests
        figure(counter)
        clf;
        counter = counter + 1;
        hold on
        plot(c_grid{end},data_e{end}{ii,jj},'k','linewidth',4);
        for kk = 1:num_resolutions
            plot(c_grid{2*kk+1},data_c{2*kk+1}{ii,jj},colors{kk},'linewidth',2);
            title(strcat(init_conds{ii},{' '},methods{jj}),'fontsize',16);
            legend('Exact', 'Computed: M = 2^{8}', 'Computed: M = 2^{10}', ...
                'Computed: M = 2^{12}', 'Computed: M = 2^{14}', ...
                'Location','Best');
        end
        hold off
        saveas(gcf,strcat('Figure_', num2str(counter-1),'.pdf'));
    end
end

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