4/12/2017
Download reference paper for thesis
Draft for the rough chapter structure

4/13/2017
Making Plans

4/19/2017
Basics of Jupyter notebook
Learn API of NumPy, Matplotlib

Inside PyCharm console
import matplotlib as mpl
import matplotlib.pyplot as plt

ScipyLectures
1. shell -> $ jupyter notebook
    ? cd run timeit debug cpaste quickref
2. NumPy
    NumPy array
        import numpy as np
        ndim shape dtype=float #default to floating point
        arrange(start, end_exclusive, step)
        linspace(start, end, num-points)
        ones() zeros() eye() diag(np.array([1,2,3,4])) tile()
            np.triu(np.ones((3,3)),1) tril()
        np.random.rand() #uniform in [0,1]
            randn() #Gaussian   seed() #set random seed
        a[2, 10] #indexing for assignment
        When modifying the view, the original array is modified as well
        Use .copy() otherwise   np.may_share_memory() can test
        Fancy indexing -> indexed with boolean or integer arrays(masks)
            -> creates copies not views
            a[a<0] = 0
    Numerical operations
        Elementwise operations
            a == b #get a boolean array for elementwise result
            np.array_equal(a, b) #array-wise comparison
            np.logical_or(a, b) logical_and()
            sin() log() exp()
            B.T #transposition for B
            Transposition is a view! Following code is wrong and will not make a matrix symmetric:
            a += a.T
        Basic reductions
            sum() axis=0/1 column/row
            min() max() argmin() argmax()
                #from numpy import unravel_index
                #unravel_index(a.argmax(), a.shape)
            all() any()
            mean() median() std()
            unique()
            np.loadtxt('filename.txt') #will chop the first line
            cumsum() #cumulative sum
            Example: 1-D Random walk
        Broadcasting
            np.arange(5)[:, np.newaxis] #get column matrix
            Broadcasting: expand matrix with grid
            Example: distance to origin
        Array shape manipulation
            a.ravel() #higher dimensions: last dimensions ravel out "first"
            b.reshape((2,3)) #inverse operation to flattening
            Reshape may or may not return a view, so for a copy use copy()
            a[:, np.newaxis]
            a = np.arange(4*3*2).reshape(4, 3, 2)
            b = a.transpose(1, 2, 0)
            Resized array cannot be referenced somewhere else
        Sorting data
            b = np.sort(a, axis=1) #sort along an axis
            a.sort(axis=1)
            j = np.argsort(a) #sort with fancy indexing: a[j] is then sorted
    More elaborate arrays
        More data types
            Mixed-type operations cast to "bigger" type
            Assignment never changes the type
            b = a.astype(int) #forced casts, truncates to integer
            b = np.around(a) #rounding, still floating-point
            int8, int16, int32, int64
            uint8,... unit64    float16,... float96, float128
            complex64,... complex256
        Structured data types
        maskedarray: dealing with missing data
    Advanced operations
        Polynomials
            p = np.poly1d([3, 2, -1]) # 3x^2 + 2x - 1
            p(0)    p.roots    p.order
            Polynomial fitting
            x = np.linspace(0, 1, 20)
            y = np.cos(x) + 0.3*np.random.rand(20)
            p = np.poly1d(np.polyfit(x, y, 3))
            t = np.linspace(0, 1, 200)
            plt.plot(x, y, 'o', t, p(t), '-')
            More polynomials
            p = np.polynomial.Polynomial([-1, 2, 3]) # 3x^2 + 2x - 1
        Loading data files
            data = np.loadtxt('data/populations.txt')
            np.savetxt('pop2.txt', data)
            Also some supports for image files

3. Matplotlib
    Introduction
        %matplotlib inline #magic for ipython
    Simple plot
        import matplotlib.pyplot as plt
        Example: sine/cosine
    Figures, Subplots, Axes and Ticks
        "figure" -> whole window in user interface    "subplots" -> within figure

    Other Types of Plots: examples and exercises
    Beyond this tutorial
    Quick references

    import matplotlib.pyplot as plt
    plt.plot(x,y,'o')
    plt.imshow(matrix_2d, cmap=plt.cm.hot)
    plt.colorbar()

4/20/2017
Learn API of Matplotlib, SciPy
Mayavi 3-D visualization
Traits: interactive dialogs
Collect background for Weak Lensing and PSF
4. SciPy

4/26/2017
Make the ellipticity chip plot:
    Make sure the definition for ellipticity(the color)
    Make sure the 36 chip relative position
Implement Polynomial, Shepard method:
    Implement Polynomial interpolation(numpy/scipy buildin)
    Implement Shepard interpolation(Look for build in/manually implement)
    Make the ellipticity chip plot

















