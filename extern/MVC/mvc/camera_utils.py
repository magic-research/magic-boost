import numpy as np
import torch


def create_camera_to_world_matrix(elevation, azimuth):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
    x = np.cos(elevation) * np.sin(azimuth)
    y = np.sin(elevation)
    z = np.cos(elevation) * np.cos(azimuth)

    # Calculate camera position, target, and up vectors
    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    # Construct view matrix
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    cam2world = np.eye(4)
    cam2world[:3, :3] = np.array([right, new_up, -forward]).T
    cam2world[:3, 3] = camera_pos
    return cam2world


def convert_opengl_to_blender(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        camera_matrix_blender = np.dot(flip_yz, camera_matrix)
    else:
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = torch.tensor(
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        )
        if camera_matrix.ndim == 3:
            flip_yz = flip_yz.unsqueeze(0)
        camera_matrix_blender = torch.matmul(flip_yz.to(camera_matrix), camera_matrix)
    return camera_matrix_blender


def normalize_camera(camera_matrix):
    """normalize the camera location onto a unit-sphere"""
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (
            np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8
        )
        camera_matrix[:, :3, 3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (
            torch.norm(translation, dim=1, keepdim=True) + 1e-8
        )
        camera_matrix[:, :3, 3] = translation
    return camera_matrix.reshape(-1, 16)


# def get_camera(
#     num_frames,
#     elevation=15,
#     azimuth_start=0,
#     azimuth_span=360,
#     blender_coord=True,
#     extra_view=False,
# ):
#     angle_gap = azimuth_span / num_frames
#     cameras = []
#     for azimuth in np.arange(azimuth_start, azimuth_span + azimuth_start, angle_gap):
#         camera_matrix = create_camera_to_world_matrix(elevation, azimuth)
#         if blender_coord:
#             camera_matrix = convert_opengl_to_blender(camera_matrix)
#         cameras.append(camera_matrix.flatten())
#
#     if extra_view:
#         dim = len(cameras[0])
#         cameras.append(np.zeros(dim))
#     return torch.tensor(np.stack(cameras, 0)).float()

def prepare_domain_embedding(domain_embedding, dtype):
    # (B, 3)
    domain_embedding = domain_embedding.to(dtype=dtype)

    domain_embedding = torch.cat([
        torch.sin(domain_embedding),
        torch.cos(domain_embedding)
    ], dim=-1)

    return domain_embedding


def prepare_relative_embedding(batch_size, cameras, can_cameras, domain_embedding, can_domain_embedding, use_domainembedding, is_v2=False):
    cameras = cameras.reshape(batch_size, -1, 3)
    can_cameras = can_cameras.reshape(batch_size, -1, 3)

    cameras = torch.cat([cameras, can_cameras], dim=1).reshape(-1, 3)
    zero_pose = can_cameras[:1, 0]
    cameras[:, 1:] = cameras[:, 1:] - zero_pose[:, 1:]

    cameras[:, 1] = torch.deg2rad(
        (torch.round(cameras[:, 1]) + 180) % 360 - 180)
    cameras[:, 0] = torch.deg2rad(cameras[:, 0])

    cameras = prepare_domain_embedding(cameras, cameras.dtype)

    domain_embedding = domain_embedding.reshape(batch_size, -1, 2)
    can_domain_embedding = can_domain_embedding.reshape(batch_size, -1, 2)
    domain_embedding = torch.cat([domain_embedding, can_domain_embedding], dim=1)
    domain_embedding = domain_embedding.reshape(-1, 2)

    if use_domainembedding:
        if is_v2:
            aug_embedding = torch.zeros_like(domain_embedding)
            # aug_embedding[:, 1] = geo_cond_strength
            domain_embedding = torch.cat([domain_embedding, aug_embedding], dim=1)
            cameras = torch.cat([domain_embedding, cameras], dim=1)
        else:
            domain_embedding = prepare_domain_embedding(domain_embedding, domain_embedding.dtype)
            cameras = torch.cat([domain_embedding, cameras], dim=1)

    # if cond_strength > 0:
    #     print('='*100)
    #     print(cond_strength)
    #     print(geo_cond_strength)

    #     if is_v2:
    #         time_step = 500 * cond_strength
    #         cameras = cameras.reshape(batch_size, -1, 10)
    #         cameras[:, 4:, 2] += time_step
    #         cameras = cameras.reshape(-1, 10)
    #     else:
    #         time_step = 500 * cond_strength
    #         noise_embedding = torch.zeros_like(can_domain_embedding)
    #         noise_embedding[:, 1:, 1] = time_step
    #         # noise_embedding = prepare_domain_embedding(noise_embedding, domain_embedding.dtype)
    #         cameras = cameras.reshape(-1, 8, 10)
    #         cameras[:, 4:, :2] += noise_embedding
    #         cameras = cameras.reshape(-1, 10)

    return cameras


def get_camera(
        num_frames,
        offset=0,
        elevation=0,
        extra_view=False,
        radius=1.5
):
    camera = []

    if not isinstance(elevation, list):
        elevation = [elevation] * num_frames
    else:
        elevation = elevation # np.random.randint(-4, 30)

    if not isinstance(offset, list):
        azimuth = [offset + i * 90 for i in range(num_frames)]
    else:
        azimuth = offset

    for i in range(num_frames):
        camera.append(np.array([elevation[i], azimuth[i], radius]))
        # azimuth = (azimuth + 90) % 360

    if extra_view:
        camera.append(np.zeros(3))
    camera = np.stack(camera, axis=0)

    camera = torch.FloatTensor(camera)
    return camera


# def prepare_relative_embedding(bs, cameras, can_cameras, domain_embedding, can_domain_embedding, use_domainembedding,  is_v2=False):
#     cameras = cameras.reshape(bs, -1, 3)
#     can_cameras = can_cameras.reshape(bs, -1, 3)

#     cameras = torch.cat([cameras, can_cameras], dim=1).reshape(-1, 3)
#     zero_pose = can_cameras[:1, 0]
#     cameras[:, 1:] = cameras[:, 1:] - zero_pose[:, 1:]

#     cameras[:, 1] = torch.deg2rad(
#         (((torch.round(cameras[:, 1])+ 360) % 360)  + 180) % 360 - 180)
#     cameras[:, 0] = torch.deg2rad(cameras[:, 0])

#     cameras = prepare_domain_embedding(cameras, cameras.dtype)

#     domain_embedding = domain_embedding.reshape(bs, -1, 2)
#     can_domain_embedding = can_domain_embedding.reshape(bs, -1, 2)
#     domain_embedding = torch.cat([domain_embedding, can_domain_embedding], dim=1)
#     domain_embedding = domain_embedding.reshape(-1, 2)

#     if use_domainembedding:
#         if is_v2:
#             aug_embedding = torch.zeros_like(domain_embedding)
#             domain_embedding = torch.cat([domain_embedding, aug_embedding], dim=1)
#             cameras = torch.cat([domain_embedding, cameras], dim=1)
#         else:
#             domain_embedding = prepare_domain_embedding(domain_embedding, domain_embedding.dtype)
#             cameras = torch.cat([domain_embedding, cameras], dim=1)
#     return cameras

def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result