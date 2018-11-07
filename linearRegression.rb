require 'sciruby'

class LinearRegression
    def cost_function(x,y,theta)
        m = y.size
        predictions = x.dot(theta)
        errors = (predictions - y)
        j = (errors*errors).sum[0,0]*1/(2*m)
        return j
    end

    def fit(x,y,theta,alpha,num_iter)
        m = y.size
        m
        j_history = N.zeros([num_iter,1])
        for i in 0..num_iter-1
            hypothesis = x.dot(theta)
            errors_vector = ((hypothesis - y).transpose).dot(x)
            mul = (alpha)*(1.0/m)
            gradient = (errors_vector.transpose)*(alpha)*(1.0/m)
            theta = theta-gradient
            j_history[i,0] = cost_function(x,y,theta)
        end
        return theta
    end
end 


test = LinearRegression.new
data = N.new([2,6], [12,15,17,18,19,20,1,2,3,4,5,6]).transpose
x = N.new([6,2])
x[:*,1] = data.col(0)
x[:*,0] = N.ones([6,1])
y = data.col(1)
theta = N.zeros([2,1])

cost = test.cost_function(x,y,theta)
grad = test.fit(x,y,theta,5,10)
pp grad
x_test = N.new([1,1],[12])
pp grad[1,0]*x_test[0,0] + grad[0,0]