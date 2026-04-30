# código para saber se a biblioteca mayavi foi instalada corretamente 

from mayavi import mlab
mlab.test_plot3d()
mlab.show()