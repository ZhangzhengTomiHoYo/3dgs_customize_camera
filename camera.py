# 2024/12/06 15:16
# Desc: 3dgs自定义视角的相机，用于自己学习3dgs的原理
# 需要在3dgs的repo里运行
import torch
import torch.nn as nn
import numpy as np
import math
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import PipelineParams

# 计算FoVx，这个函数不需要写到类里
def compute_fovX(fovY, height, width):
    w_h_ratio = float(width) / float(height)

    return math.atan(math.tan(fovY * 0.5) * w_h_ratio) * 2

# view矩阵就是w2c矩阵
def getViewMatrix(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.T
    Rt[:3, 3] = -R.T @ t
    Rt[3, 3] = 1.0

    return torch.tensor(Rt, dtype=torch.float32)

# 投影矩阵，就是c2w矩阵？ 好吧，其实我不是很懂这个矩阵
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class VirtualCamera(nn.Module):
    def __init__(self):
        # 旋转矩阵R，还没学会其中的原理，只知道这个单位矩阵是和世界坐标系一致
        self.R = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]
        )
        # 平移矩阵T，这个好理解，就是相机坐标系的原点在世界坐标系的位置
        self.T = np.array(
            [
                [0, 0, -2] #表示沿z轴负方向移动2个单位
            ]
        )
        # 相机的那个视野大小
        self.FoVy = math.radians(30)
        # 
        self.image_height =1024
        #
        self.image_width = 1024
        # 得到FoVx
        self.FoVx = compute_fovX(self.FoVy, self.image_height, self.image_width)
        # 
        self.zfar = 100.0
        # 等价于相机的焦距
        self.znear = 0.1
        # w2c
        self.world_view_transform = getViewMatrix(self.R, self.T)

        # TODO: 复制粘贴了下面这些东西，但不知道原理
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        )
        self.full_proj_transform = self.projection_matrix @ self.world_view_transform

        self.world_view_transform = self.world_view_transform.transpose(0, 1).cuda()
        self.projection_matrix = self.projection_matrix.transpose(0, 1).cuda()
        self.full_proj_transform = self.full_proj_transform.transpose(0, 1).cuda()

        self.camera_center = torch.tensor(self.T, dtype=torch.float32).cuda()
        
        # 仅需要相机位置和高斯点云/目标物体的中心位置就可以确定相机视角的函数，非常的厉害，不用自己写R矩阵
        def setLookAt(self, camera_position, target_position):
            cam_pos = np.array(cam_pos, dtype=np.float32)
            target_pos = np.array(target_pos, dtype=np.float32)
            y_approx = np.array([0, 1, 0], dtype=np.float32)
            look_dir = target_position - camera_position
            z_axis = look_dir
            z_axis /= np.linalg.norm(z_axis)
            x_axis = np.cross(y_approx, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            y_axis /= np.linalg.norm(y_axis)
            R = np.zeros((3, 3))
            R[:, 0] = x_axis
            R[:, 1] = y_axis
            R[:, 2] = z_axis
            self.R = R
            self.T = camera_position

            # 必须要重新更新一些矩阵
            self.world_view_transform = getViewMatrix(self.R, self.T)
            self.projection_matrix = getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            self.full_proj_transform = self.projection_matrix @ self.world_view_transform
            self.world_view_transform = self.world_view_transform.transpose(0, 1).cuda()
            self.projection_matrix = self.projection_matrix.transpose(0, 1).cuda()
            self.full_proj_transform = self.full_proj_transform.transpose(0, 1).cuda()
            self.camera_center = torch.tensor(self.T, dtype=torch.float32).cuda()

    # 不懂，用gpt注释了一下
    def set_theta_phi_distance(self, theta, phi, distance):
        if theta > 360 or theta < 0:  # 如果theta大于360或小于0
            theta = theta % 360  # 将theta限制在0到360之间
        theta = math.radians(theta)  # 将theta转换为弧度
        phi = math.radians(phi)  # 将phi转换为弧度
        cam_pos = [
            math.cos(phi) * math.cos(theta) * distance,  # 计算相机x坐标
            math.sin(phi) * distance,  # 计算相机y坐标
            math.cos(phi) * math.sin(theta) * distance,  # 计算相机z坐标
        ]
        self.setLookAt(cam_pos, target_pos=[0, 0, 0])  # 设置相机视角，目标位置为原点

@torch.no_grad
def test_virtual_camera(
    fov=10,
    image_width=512,
    image_height=512,
    distance=1.1,
    theta=0,
    phi=0,
    r=0,
    g=0,
    b=0,
):
    # 空的，因为调用render必须要这个参数
    parser = ArgumentParser(description="gaussian splatting")
    pipe = PipelineParams(parser)

    gaussians = GaussianModel(3)
    gaussians.load_ply("./models/test_faceca.ply")
    # TODO 看来人家把东西写到形参里是对的，我那样写后面初始化的时候没办法自定义了
    camera = VirtualCamera()
    # 球坐标，可以代替setLookAt函数，不会用，也没必要
    # camera.set_theta_phi_distance(theta, phi, distance)
    bg = torch.tensor([r, g, b], dtype=torch.float32, device="cuda")
    image = render(camera, gaussians, pipe, bg)["render"]
    torchvision.utils.save_image(image, "./debug/mycamera.png")

if __name__ == "__main__":
    test_virtual_camera()