from pyvista import demos


def add_origin_cube(plotter):
    ocube = demos.orientation_cube()
    plotter.add_mesh(ocube["cube"], show_edges=True)
    plotter.add_mesh(ocube["x_p"], color="blue")
    plotter.add_mesh(ocube["x_n"], color="blue")
    plotter.add_mesh(ocube["y_p"], color="green")
    plotter.add_mesh(ocube["y_n"], color="green")
    plotter.add_mesh(ocube["z_p"], color="red")
    plotter.add_mesh(ocube["z_n"], color="red")
