import numpy as np
import my_utils
import cv2
import numpy as np

from scipy.spatial import ConvexHull
"""
# Camera poses (rotation and translation relative to the world coordinates)
R1 = np.array([...])  # 3x3 rotation matrix
T1 = np.array([...])  # 3x1 translation vector
R2 = np.array([...])  # 3x3 rotation matrix
T2 = np.array([...])  # 3x1 translation vector

# Intrinsic matrix
K = np.array([...])   # 3x3 camera calibration matrix

# Image size
H, W = (480, 640)     # Height and width of the images
"""
def calculate_overlap(R1,T1,R2,T2,K,H,W):
    # Define the camera matrices
    P1 = K @ np.hstack((R1, T1))
    P2 = K @ np.hstack((R2, T2))


    # Define the image grids for the two cameras
    x = np.arange(W)
    y = np.arange(H)
    xx, yy = np.meshgrid(x, y)

    # Flatten the image grids into homogeneous coordinates
    pts1 = np.column_stack((xx.flatten(), yy.flatten(), np.ones((H*W,))))
    pts2 = np.column_stack((xx.flatten(), yy.flatten(), np.ones((H*W,))))

    # Compute the projection matrices for the two cameras
    # P1_inv = np.linalg.inv(P1)
    # P2_inv = np.linalg.inv(P2)
    P1_hom = np.vstack((P1, [0, 0, 0, 1]))
    P1_inv_hom = np.linalg.inv(P1_hom)
    P1_inv = P1_inv_hom[:3, :4]
    
    P2_hom = np.vstack((P2, [0, 0, 0, 1]))
    P2_inv_hom = np.linalg.inv(P2_hom)
    P2_inv = P2_inv_hom[:3, :4]



    border1 = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]])
    border2 = np.array([[W//2, 0], [W-1, 0], [W-1, H-1], [W//2, H-1]])
    # Transform image boundaries to world coordinates
    border1_world = P1_inv.T @ np.hstack((border1, np.ones((4, 1)))).T
    border1_world /= border1_world[3, :]
    border1_world = border1_world[:3, :].T

    border2_world = P2_inv.T @ np.hstack((border2, np.ones((4, 1)))).T
    border2_world /= border2_world[3, :]
    border2_world = border2_world[:3, :].T

    # Find overlap in world coordinates
    # overlap_world = np.intersect1d(border1_world[:, 0], border2_world[:, 0])

    # # Transform overlap to image coordinates
    # overlap1 = (P1 @ np.hstack((overlap_world, np.zeros((len(overlap_world), 1)), np.ones((len(overlap_world), 1)))).T)[:2, :]
    # overlap2 = (P2 @ np.hstack((overlap_world, np.zeros((len(overlap_world), 1)), np.ones((len(overlap_world), 1)))).T)[:2, :]
    # # Create masks for common region in the two images
    # mask1 = np.zeros((H, W), dtype=np.uint8)
    # mask2 = np.zeros((H, W), dtype=np.uint8)

    # mask1[(overlap1[1, :]).astype(int), (overlap1[0, :]).astype(int)] = 1
    # mask2[(overlap2[1, :]).astype(int), (overlap2[0, :]).astype(int)] = 1
    # if False:
    #     # Transform the image points using the camera projection matrices
    #     pts1 = P1_inv.T @ pts1.T
    #     pts2 = P2_inv.T @ pts2.T

    #     # Normalize the homogeneous coordinates
    #     pts1 = pts1 / pts1[-1,:]
    #     pts2 = pts2 / pts2[-1,:]

    #     # Calculate the Euclidean distance between the transformed image points
    #     # dist = np.linalg.norm(pts1[:2,:] - pts2[:2,:], axis=0)
    #     # Calculate distance between transformed points
    #     dist = np.linalg.norm(pts1[:2,:,None] - pts2[:2,None,:], axis=0)


    #     # Define a threshold for the distance (e.g. 5 pixels)
    #     threshold = 5

    #     # Create the masks
    #     mask1 = np.reshape(dist <= threshold, (H, W))
    #     mask2 = np.reshape(dist <= threshold, (H, W))
    #     return mask1,mask2
    # Compute the convex hull of the eight resulting points
    points = np.vstack((border1_world, border2_world))
    hull = ConvexHull(points)    
    
    # Transform the convex hull back to the camera frame to obtain the 3D box that is common to both images
    hull_world = points[hull.vertices].T
    hull_camera1 = P1 @ np.vstack((hull_world, np.ones((1, hull_world.shape[1]))))
    hull_camera2 = P2 @ np.vstack((hull_world, np.ones((1, hull_world.shape[1]))))
    # Compute the bounding box of the common region in image coordinates
    x_min = int(np.min(hull_camera1[:, 0]))
    y_min = int(np.min(hull_camera1[:, 1]))
    x_max = int(np.max(hull_camera1[:, 0]))
    y_max = int(np.max(hull_camera1[:, 1]))
    bbox1 = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

    x_min = int(np.min(hull_camera2[:, 0]))
    y_min = int(np.min(hull_camera2[:, 1]))
    x_max = int(np.max(hull_camera2[:, 0]))
    y_max = int(np.max(hull_camera2[:, 1]))
    bbox2 = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

    # Initialize the masks
    mask1 = np.zeros((H,W))
    mask2 = np.zeros((H,W))

    # Get the coordinates of the bounding box for image 1
    x1, y1, w1, h1 = bbox1

    # Set the pixels inside the bounding box to 255 in mask1
    mask1[y1:y1+h1, x1:x1+w1] = 255

    # Get the coordinates of the bounding box for image 2
    x2, y2, w2, h2 = bbox2

    # Set the pixels inside the bounding box to 255 in mask2
    mask2[y2:y2+h2, x2:x2+w2] = 255


def test():


    # Define camera intrinsic matrix K
    K = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]])

    # Define image size
    H, W = 224,224

    # Calculate extrinsic matrix for camera 1
    azim1, elev1, dist1 = 0, 0, 5
    R1, _ = cv2.Rodrigues(np.array([azim1, elev1, 0.]))
    T1 = np.array([0, 0, dist1]).reshape(3,1)

    # Calculate extrinsic matrix for camera 2
    azim2, elev2, dist2 = 0, 15, 5
    R2, _ = cv2.Rodrigues(np.array([azim2, elev2, 0.]))
    T2 = np.array([0, 0, dist2]).reshape(3,1)

    # Define image overlap
    overlap_H, overlap_W = H, W//2

    # Calculate translation between cameras to achieve overlap
    delta_x = ((W-overlap_W)/2)/K[0,0] * dist1
    delta_z = (W-overlap_W)/2/K[0,0] * dist1

    # Update camera poses
    T1[0,0] += delta_x
    T2[0,0] -= delta_x
    T1[2,0] -= delta_z
    T2[2,0] -= delta_z
    mask1,mask2 = calculate_overlap(R1,T1,R2,T2,K,H,W)
if __name__ == '__main__':
    test()