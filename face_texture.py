import eos
import cv2
class FaceTexture:
    def __init__(self, files):
        self.stopped = False
        (self.pts, self.image_path) = files
        self.main()

    def main(self):
        landmarks = self.read_pts(self.pts)
        image = cv2.imread(self.image_path)

        # height and width using opencv2
        h,w = image.shape[:2]

        # eos example file
        model = eos.morphablemodel.load_model("share/sfm_shape_3448.bin")
        blendshapes = eos.morphablemodel.load_blendshapes("share/expression_blendshapes_3448.bin")

        # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
        morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                            color_model=eos.morphablemodel.PcaModel(),
                                                                            vertex_definitions=None,
                                                                            texture_coordinates=model.get_texture_coordinates())
        
        landmark_mapper = eos.core.LandmarkMapper('share/ibug_to_sfm.txt')
        edge_topology = eos.morphablemodel.load_edge_topology('share/sfm_3448_edge_topology.json')
        contour_landmarks = eos.fitting.ContourLandmarks.load('share/ibug_to_sfm.txt')
        model_contour = eos.fitting.ModelContour.load('share/sfm_model_contours.json')

        (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(morphablemodel_with_expressions,
            landmarks, landmark_mapper, w, h, edge_topology, contour_landmarks, model_contour)

        # Now you can use your favourite plotting/rendering library to display the fitted mesh, using the rendering
        # parameters in the 'pose' variable.

        # Or for example extract the texture map, like this:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA, 4)  # extract_texture(...) expects a 4-channel image
        texturemap = eos.render.extract_texture(mesh, pose, image)
        cv2.imwrite("texture.png",texturemap)
        eos.core.write_obj(mesh,"mesh.obj")
        self.stop()

    def stop(self):
        self.stopped = True

    def read_pts(self,filename):
        """A helper function to read the 68 ibug landmarks from a .pts file."""
        lines = open(filename).read().splitlines()
        lines = lines[3:71]

        landmarks = []
        ibug_index = 1  # count from 1 to 68 for all ibug landmarks
        for l in lines:
            coords = l.split()
            landmarks.append(eos.core.Landmark(str(ibug_index), [float(coords[0]), float(coords[1])]))
            ibug_index = ibug_index + 1

        return landmarks

if __name__ == "__main__":
   FaceTexture(("image/pts.pts","image/face_0.png"))
