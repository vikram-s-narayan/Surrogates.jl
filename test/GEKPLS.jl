using Surrogates
using Zygote


function water_flow(x)
    r_w = x[1]
    r = x[2]
    T_u = x[3]
    H_u = x[4]
    T_l = x[5]
    H_l = x[6]
    L = x[7]
    K_w = x[8]
    log_val = log(r/r_w)
    return (2*pi*T_u*(H_u - H_l))/ ( log_val*(1 + (2*L*T_u/(log_val*r_w^2*K_w)) + T_u/T_l))
end




function vector_of_tuples_to_matrix(v)
    #convert training data generated by surrogate sampling into a matrix suitable for GEKPLS
    num_rows = length(v)
    num_cols = length(first(v))
    K = zeros(num_rows, num_cols)
    for row in 1:num_rows
        for col in 1:num_cols
            K[row, col]=v[row][col]
        end
    end
    return K
end

function vector_of_tuples_to_matrix2(v)
    #convert gradients into matrix form
    num_rows = length(v)
    num_cols = length(first(first(v)))
    K = zeros(num_rows, num_cols)
    for row in 1:num_rows
        for col in 1:num_cols
            K[row, col] = v[row][1][col]
        end
    end
    return K
end

n = 1000
d = 8
lb = [0.05,100,63070,990,63.1,700,1120,9855]
ub = [0.15,50000,115600,1110,116,820,1680,12045]
x = sample(n,lb,ub,SobolSample())
X = vector_of_tuples_to_matrix(x)
grads = vector_of_tuples_to_matrix2(gradient.(water_flow, x))
y = reshape(water_flow.(x),(size(x,1),1))
xlimits = hcat(lb, ub)
n_test = 100 
x_test = sample(n_test,lb,ub,GoldenSample()) 
X_test = vector_of_tuples_to_matrix(x_test) 
y_true = water_flow.(x_test)

@testset "Water Flow Function Test (dimensions = 8; n_comp = 2; extra_points = 2; initial_theta=.01)" begin 
    n_comp = 2
    delta_x = 0.0001
    extra_points = 2
    initial_theta = 0.01
    g = GEKPLS(X, y, grads, n_comp, delta_x, xlimits, extra_points, initial_theta) #change hard-coded 2 param to variable
    y_pred = g(X_test)
    rmse = sqrt(sum(((y_pred - y_true).^2)/n_test))
    println("rmse: ", rmse) #rmse: 0.028999713540341848
    @test isapprox(rmse, 0.02, atol=0.01)
end

@testset "Water Flow Function Test (dimensions = 8; n_comp = 3; extra_points = 2; initial_theta=.01)" begin 
    n_comp = 3
    delta_x = 0.0001
    extra_points = 2
    initial_theta = 0.01
    g = GEKPLS(X, y, grads, n_comp, delta_x, xlimits, extra_points, initial_theta) #change hard-coded 2 param to variable
    y_pred = g(X_test)
    rmse = sqrt(sum(((y_pred - y_true).^2)/n_test))
    println("rmse: ", rmse) #rmse: 0.024098165334772537
    @test isapprox(rmse, 0.02, atol=0.01)
end

@testset "Water Flow Function Test (dimensions = 8; n_comp = 3; extra_points = 3; initial_theta=.01)" begin 
    n_comp = 3
    delta_x = 0.0001
    extra_points = 3
    initial_theta = 0.01
    g = GEKPLS(X, y, grads, n_comp, delta_x, xlimits, extra_points, initial_theta) #change hard-coded 2 param to variable
    y_pred = g(X_test)
    rmse = sqrt(sum(((y_pred - y_true).^2)/n_test))
    println("rmse: ", rmse) #0.023712136706924902
    @test isapprox(rmse, 0.02, atol=0.01)
end