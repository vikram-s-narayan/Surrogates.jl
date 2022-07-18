using Surrogates
using Zygote

function sphere_function(x)
    return sum(x .^ 2)
end

n = 30
d = 3
lb = [-10.0 for i in 1:d]
ub = [10.0 for i in 1:d]
x = sample(n, lb, ub, SobolSample())
grads = Zygote.gradient.(sphere_function, x)
y = sphere_function.(x)
xlimits = hcat(lb, ub)

n_test = 20
x_test = sample(n_test, lb, ub, GoldenSample())
y_true = sphere_function.(x_test)
n_comp = 2
delta_x = 0.0001
extra_points = 2
initial_theta = [0.01 for i in 1:n_comp]

g = GEKPLS(x, y, grads, n_comp, delta_x, xlimits, extra_points, initial_theta)
sphere_function.([(1.0, 1.0, 1.0)])
g([(1.0, 1.0, 1.0)])
my_point = [(1.0, 1.0, 1.0)]
g(my_point)
x_test
g(x_test)
#gradient(g, [1.0 1.0 1.0])
b = [(2.0, 5.0, 5.0)]
a = (2.0, 5.0, 5.0)
Zygote.gradient(g, a) 
Zygote.jacobian(g, a)
Zygote.gradient(g, b)




# typeof(a)
# size(a);


# my_rad_SRBFN = RadialBasis(x, sphere_function.(x), lb, ub, rad = linearRadial())
# my_rad_SRBFN.(my_point)
# #my_rad_SRBFN'(my_point)
# new_rad = x -> Zygote.gradient(my_rad_SRBFN, x)
# new_rad((2.0, 5.0, 5.0))
# my_rad_SRBFN(a)

# Zygote.gradient.(my_rad_SRBFN, b)
# Zygote.gradient.(g, b)
# g(b)
# g.(b)
# surrogate_optimize(sphere_function, SRBF(), lb, ub, my_rad_SRBFN, UniformSample())

# surrogate_optimize(sphere_function, SRBF(), lb, ub, g, UniformSample())

# ###
# my_rad_SRBFN.x
# g.x