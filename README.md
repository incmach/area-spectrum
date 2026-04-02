This is a program for computing "area spectrum transform" (AST for short) of an image, modifying this spectrum and fitting an image to the modified version (computing "inverse area spectrum" transform, IAST).

Here
an _image_ is a multiset of points on a plane --- given either as a pixel grayscale image where a pixel value gives the multiplicty of a point at its coordinate or simply a list of points,
an _area spectrum_ is a multiset of areas of triangles with vertices at the image's points --- given either as a vector where each value gives the number of triangles with area half of its index or a vector of values of symmetric functions of areas.

The purpose is to research such "geometric" transforms and have some fun with pictures.
