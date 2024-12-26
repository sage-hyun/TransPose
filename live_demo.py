# python 11 fix bug
import inspect
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
    
import socket
import threading
from articulate.math import *
from datetime import datetime
import torch
import numpy as np
import config
import time
from net import TransPoseNet
from pygame.time import Clock
import socketio
from utils import normalize_and_concat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inertial_poser = TransPoseNet(num_past_frame=20, num_future_frame=5).to(device)
running = False
start_recording = False


class IMUSet:
    r"""
    Sensor order: left forearm, right forearm, left lower leg, right lower leg, head, pelvis
    """
    def __init__(self, imu_host='127.0.0.1', imu_port=7002, buffer_len=26):
        """
        Init an IMUSet for Noitom Perception Legacy IMUs. Please follow the instructions below.

        Instructions:
        --------
        1. Start `Axis Legacy` (Noitom software).
        2. Click `File` -> `Settings` -> `Broadcasting`, check `TCP` and `Calculation`. Set `Port` to 7002.
        3. Click `File` -> `Settings` -> `Output Format`, change `Calculation Data` to
           `Block type = String, Quaternion = Global, Acceleration = Sensor local`
        4. Place 1 - 6 IMU on left lower arm, right lower arm, left lower leg, right lower leg, head, root.
        5. Connect 1 - 6 IMU to `Axis Legacy` and continue.

        :param imu_host: The host that `Axis Legacy` runs on.
        :param imu_port: The port that `Axis Legacy` runs on.
        :param buffer_len: Max number of frames in the readonly buffer.
        """
        self.imu_host = imu_host
        self.imu_port = imu_port
        self.clock = Clock()

        self._imu_socket = None
        self._buffer_len = buffer_len
        self._ori_buffer = []
        self._acc_buffer = []
        self._is_reading = False
        self._read_thread = None

    def _read(self):
        """
        The thread that reads imu measurements into the buffer. It is a producer for the buffer.
        """
        num_float_one_frame = 6 * 12
        data = ''
        while self._is_reading:
            msg, clientAddr = self._imu_socket.recvfrom(8192)
            data = msg.decode('ascii')

            for one_frame in data.split("#"):
                strs = one_frame.split(' ', num_float_one_frame)

                # print(len(strs))
                # print(len(data.split(' ')))

                acc = np.array(strs[:6*3]).reshape(6,3).astype(float)
                ori = np.array(strs[6*3:]).reshape(6,3,3).astype(float)

                if len(strs) >= num_float_one_frame:
                    # print(np.array(strs[:-3]).reshape((21, 16)))  # full data
                    # d = np.array(strs[:96]).reshape((6, 16))  # first 6 imus
                    tranc = int(len(self._ori_buffer) == self._buffer_len)
                    self._ori_buffer = self._ori_buffer[tranc:] + [ori]
                    self._acc_buffer = self._acc_buffer[tranc:] + [acc]
                    # data = strs[-1]
                    self.clock.tick(60)

    def start_reading(self):
        """
        Start reading imu measurements into the buffer.
        """
        if self._read_thread is None:
            self._is_reading = True
            self._read_thread = threading.Thread(target=self._read)
            self._read_thread.setDaemon(True)
            self._ori_buffer = []
            self._acc_buffer = []
            self._imu_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # self._imu_socket.connect((self.imu_host, self.imu_port))
            self._imu_socket.bind(('', self.imu_port))
            self._read_thread.start()
        else:
            print('Failed to start reading thread: reading is already start.')

    def stop_reading(self):
        """
        Stop reading imu measurements.
        """
        if self._read_thread is not None:
            self._is_reading = False
            self._read_thread.join()
            self._read_thread = None
            self._imu_socket.close()

    def get_current_buffer(self):
        """
        Get a view of current buffer.

        :return: Quaternion and acceleration torch.Tensor in shape [buffer_len, 6, 4] and [buffer_len, 6, 3].
        """
        q = torch.tensor(self._ori_buffer, dtype=torch.float)
        a = torch.tensor(self._acc_buffer, dtype=torch.float)
        return q, a

    def get_mean_measurement_of_n_second(self, num_seconds=3, buffer_len=120):
        """
        Start reading for `num_seconds` seconds and then close the connection. The average of the last
        `buffer_len` frames of the measured quaternions and accelerations are returned.
        Note that this function is blocking.

        :param num_seconds: How many seconds to read.
        :param buffer_len: Buffer length. Must be smaller than 60 * `num_seconds`.
        :return: The mean quaternion and acceleration torch.Tensor in shape [6, 4] and [6, 3] respectively.
        """
        save_buffer_len = self._buffer_len
        self._buffer_len = buffer_len
        self.start_reading()
        time.sleep(num_seconds)
        self.stop_reading()
        q, a = self.get_current_buffer()
        self._buffer_len = save_buffer_len

        print(len(self._ori_buffer))
        return q.mean(dim=0), a.mean(dim=0)


def get_input():
    global running, start_recording
    while running:
        c = input()
        if c == 'q':
            running = False
        elif c == 'r':
            start_recording = True
        elif c == 's':
            start_recording = False


if __name__ == '__main__':
    imu_set = IMUSet(buffer_len=1)

    # input('Put imu 1 aligned with your body reference frame (x = Left, y = Up, z = Forward) and then press any key.')
    # print('Keep for 3 seconds ...', end='')
    # oris = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)[0][0]
    # smpl2imu = quaternion_to_rotation_matrix(oris).view(3, 3).t()

    # input('\tFinish.\nWear all imus correctly and press any key.')
    # for i in range(3, 0, -1):
    #     print('\rStand straight in T-pose and be ready. The celebration will begin after %d seconds.' % i, end='')
    #     time.sleep(1)
    # print('\rStand straight in T-pose. Keep the pose for 3 seconds ...', end='')
    # oris, accs = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=200)
    # oris = quaternion_to_rotation_matrix(oris)
    # device2bone = smpl2imu.matmul(oris).transpose(1, 2).matmul(torch.eye(3))
    # acc_offsets = smpl2imu.matmul(accs.unsqueeze(-1))   # [num_imus, 3, 1], already in global inertial frame
    
    # print('\tFinish.\nStart estimating poses. Press q to quit, r to record motion, s to stop recording.')

    # server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_for_unity.bind(('127.0.0.1', 7004))
    # server_for_unity.listen(1)
    # print('Server start. Waiting for unity3d to connect.')
    # conn, addr = server_for_unity.accept()

    imu_set.start_reading()

    # unity_client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # unity_client_socket.connect(('127.0.0.1', 7004))

    sio = socketio.Client()
    sio.connect('http://143.248.143.65:5555')

    # 서버로부터 이벤트를 받는 핸들러
    @sio.on('start_sending')
    def handle_start_sending():
        global running, inertial_poser
        running = True
        inertial_poser = TransPoseNet(num_past_frame=20, num_future_frame=5).to(device)
        imu_set.start_reading()
        print('start_sending event received.')

    @sio.on('stop_sending')
    def handle_stop_sending():
        global running
        running = False
        imu_set.stop_reading()
        print('stop_sending event received.')

    running = True
    clock = Clock()
    is_recording = False
    record_buffer = None

    get_input_thread = threading.Thread(target=get_input)
    get_input_thread.setDaemon(True)
    get_input_thread.start()

    import onnxruntime as ort
    # ONNX 모델 실행 준비
    session = ort.InferenceSession("transpose_net_241226_opset17.onnx")

    h_state = torch.zeros(2, 256)
    c_state = torch.zeros(2, 256)
    root_y = torch.tensor(0.0).view(1,1)
    lfoot_pos = torch.tensor([ 0.1283, -0.9559,  0.0750])
    rfoot_pos = torch.tensor([-0.1194, -0.9564,  0.0774])
    tran = torch.zeros(3)
    
    h_state_2, c_state_2, root_y_2, lfoot_pos_2, rfoot_pos_2, tran_2 = h_state, c_state, root_y, lfoot_pos, rfoot_pos, tran
    h_state, c_state, root_y, lfoot_pos, rfoot_pos, tran = h_state.numpy(), c_state.numpy(), root_y.numpy(), lfoot_pos.numpy(), rfoot_pos.numpy(), tran.numpy()
    
    pose_1 = None
    tran_1 = None

    while True:
        while running:
            # calibration
            clock.tick(60)
            # ori_raw, acc_raw = imu_set.get_current_buffer()   # [1, 6, 4], get measurements in running fps
            # ori_raw = quaternion_to_rotation_matrix(ori_raw).view(1, 6, 3, 3)
            # acc_cal = (smpl2imu.matmul(acc_raw.view(-1, 6, 3, 1)) - acc_offsets).view(1, 6, 3)
            # ori_cal = smpl2imu.matmul(ori_raw).matmul(device2bone)


            ori, acc = imu_set.get_current_buffer()

            # normalization
            # acc = torch.cat((acc_cal[:, :5] - acc_cal[:, 5:], acc_cal[:, 5:]), dim=1).bmm(ori_cal[:, -1]) / config.acc_scale
            # ori = torch.cat((ori_cal[:, 5:].transpose(2, 3).matmul(ori_cal[:, :5]), ori_cal[:, 5:]), dim=1)
            
            if not len(acc) and not len(ori):
                continue

            print(f"\n{acc.shape=}")
            print(f"{ori.shape=}")

            # data_nn = torch.cat((acc.view(-1, 6*3), ori.view(-1, 6*9)), dim=1).to(device)
            
            # data_nn = normalize_and_concat(acc, ori).to(device)
            # pose, tran = inertial_poser.forward_online(data_nn)
            # pose = rotation_matrix_to_axis_angle(pose.view(1, 216)).view(72)

            # pose, tran, h_state, c_state, root_y, lfoot_pos, rfoot_pos = inertial_poser.forward(acc, ori, 
            #                                     h_state,
            #                                     c_state,
            #                                     root_y,
            #                                     lfoot_pos,
            #                                     rfoot_pos,
            #                                     tran)
            pose_2, tran_2, h_state_2, c_state_2, root_y_2, lfoot_pos_2, rfoot_pos_2 = inertial_poser.forward(acc, ori, 
                                                h_state_2,
                                                c_state_2,
                                                root_y_2,
                                                lfoot_pos_2,
                                                rfoot_pos_2,
                                                tran_2)
            
            input_feed = {
                "acc": acc.view(1,18).numpy(),
                "ori": ori.view(1,54).numpy(),
                "h_state": h_state,
                "c_state": c_state,
                "current_root_y": root_y,
                "last_lfoot_pos": lfoot_pos,
                "last_rfoot_pos": rfoot_pos,
                "last_tran": tran,
            }
            pose, tran, h_state, c_state, root_y, lfoot_pos, rfoot_pos = session.run(["pose", "tran", "h", "c", "root_y", "lfoot_pos", "rfoot_pos"], input_feed)

            print(lfoot_pos, lfoot_pos_2)
            print(rfoot_pos, rfoot_pos_2)
            # if tran_1 is None:
            #     tran_1 = tran

            # # recording
            # if not is_recording and start_recording:
            #     record_buffer = data_nn.view(1, -1)
            #     is_recording = True
            # elif is_recording and start_recording:
            #     record_buffer = torch.cat([record_buffer, data_nn.view(1, -1)], dim=0)
            # elif is_recording and not start_recording:
            #     torch.save(record_buffer, 'data/imu_recordings/r' + datetime.now().strftime('%T').replace(':', '-') + '.pt')
            #     is_recording = False

            # send pose
            s = ','.join(['%g' % v for v in pose]) + '#' + \
                ','.join(['%g' % v for v in tran]) + '$'
            # print("s:", s)
            # unity_client_socket.send(s.encode('utf8'))  # I use unity3d to read pose and translation for visualization here
            sio.emit('animation_data', s)
            # sio.sleep(1 / 60)


            print('\r', '(recording)' if is_recording else '', 'Sensor FPS:', round(imu_set.clock.get_fps(),2),
                '\tOutput FPS:', round(clock.get_fps(),2), end='\t')

        # finally:
        #     unity_client_socket.close()
        #     print("unity_client_socket closed")

        # get_input_thread.join()
        # imu_set.stop_reading()
