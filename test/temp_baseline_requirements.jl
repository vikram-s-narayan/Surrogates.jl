using Surrogates
using Zygote

#sphere function
# function s(x)
#     return sum(x .^ 2)
# end

# n = 30
# d = 3
# lb = [-10.0 for i in 1:d]
# ub = [10.0 for i in 1:d]
# x = sample(n, lb, ub, SobolSample())
# y = s.(x)

# r = RadialBasis(x, s.(x), lb, ub, rad = linearRadial())
# s([(1.0, 1.0, 1.0)])
# s([(1.0, 1.0, 1.0)])
# s((1.0, 1.0, 1.0))
# r((1.0, 1.0, 1.0))

# test_x = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
# s.(test_x)
# r.(test_x)

# gradient(s, [(1.0, 1.0, 1.0)])
# gradient(r, [(1.0, 1.0, 1.0)])

# gradient(s, (1.0, 1.0, 1.0))
# gradient(r, (1.0, 1.0, 1.0))

# gradient.(s, test_x)
# gradient.(r,test_x)

#surrogate_optimize(s, SRBF(), lb, ub, r, UniformSample(); maxiters = 10, num_new_samples = 10, needs_gradient=true)

##### above are our baseline requirments
#### below then are experiments with gekpls
function s(x)
    return sum(x .^ 2)
end

n = 30
d = 3
lb = [-10.0 for i in 1:d]
ub = [10.0 for i in 1:d]
x = sample(n, lb, ub, SobolSample())
y = s.(x)
n_comp = 2
delta_x = 0.0001
extra_points = 2
initial_theta = [0.01 for i in 1:n_comp]
grads = Zygote.gradient.(s, x)
xlimits = hcat(lb, ub)
g = GEKPLS(x, y, grads, n_comp, delta_x, xlimits, extra_points, initial_theta)

# add_point!(g, (1.0, 1.0, 1.0), 3.0, (2.0, 2.0, 2.0))
# g.x
# g.x_matrix
# size(g.y)
# g.grads
# g.y

g((1.0, 1.0, 1.0)) #works as expected - similar to a regular function
test_x = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
g.(test_x) #works fine!! 
g(test_x) #so this is not supposed to work fine but it still does any way :)
gradient(g, (1.0, 1.0, 1.0)) #works fine just like a regular function!
gradient.(g, test_x) # works fine


one_tup_vec = [(1.0, 1.0, 1.0)]
g.(one_tup_vec) #works fine


surrogate_optimize(s, SRBF(), lb, ub, g, UniformSample(); needs_gradient=true)

g.x