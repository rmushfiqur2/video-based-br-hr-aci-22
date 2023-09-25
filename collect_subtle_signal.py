import os
import seaborn as sns

from motion_vector.GMA.core.utils.frame_utils import *
from motion_vector.GMA.core.utils.flow_viz import *
import cv2

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from math import floor
from scipy.interpolate import griddata, interp2d
import json

# https://stackoverflow.com/questions/45729092/
# make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7/45734500#45734500
import matplotlib

matplotlib.use("TkAgg")


def mypause(interval):
    backend = plt.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def update_rect(points, rect):
    x0, y0 = round(np.min(points[:, 0])), round(np.min(points[:, 1]))
    x1, y1 = round(np.max(points[:, 0])), round(np.max(points[:, 1]))
    rect.set_xy((x0, y0))
    rect.set_width(x1 - x0)
    rect.set_height(y1 - y0)


class Annotate(object):

    def __init__(self,
                 flo_file,
                 img_grid_sig_cap,
                 forced_size=None,
                 interp=None,
                 num_patch=7,
                 vis_on=True,
                 calc_on=True,
                 vis_res=5,
                 calc_res=1,
                 vis_radius=2,
                 first_frame=None,
                 plot_signal=False,
                 rect_move=False,
                 crop=None,
                 record='displacement',
                 global_ref=False):
        assert (record == 'displacement' or record
                == 'rgb'), "only mv or rgb can be the type of signal recorded"
        self.global_ref = global_ref
        self.pressed = False
        self.on_img_grid = img_grid_sig_cap
        self.forced_size = forced_size
        self.rect_move = rect_move
        self.crop = crop
        self.plot_signal = plot_signal
        self.record = record
        self.org_video = self.record == 'rgb'

        if self.plot_signal:
            self.fig, self.axes = plt.subplots(nrows=2,
                                               ncols=2,
                                               figsize=(12, 6),
                                               dpi=100)
            self.ax = self.axes[0][0]  # axes[0] is nrows=1 or ncols=1
            self.ax_u = self.axes[1][1]
            self.ax_v = self.axes[1][0]
            self.ax_flo = self.axes[0][1]
        else:
            self.fig, self.axes = plt.subplots(nrows=1,
                                               ncols=1,
                                               figsize=(12, 6),
                                               dpi=100)
            self.ax = self.axes
        self.fig.subplots_adjust(wspace=0.5)
        self.num_patch = num_patch
        self.clrs = sns.color_palette('husl', n_colors=self.num_patch)
        self.rect = []
        self.points = []
        self.points_ini = []  # initial points
        self.points_vis = []  # [[x,y],..] numpy
        self.points_vis_ini = []  # [[x,y],..] numpy (initial points)
        self.points_vis_obj = []  # matplotlib object
        self.vis_on = vis_on
        self.calc_on = calc_on
        self.points_res = calc_res
        self.points_vis_res = vis_res
        self.points_vis_radius = vis_radius
        self.interpolation = interp  # 'linear' / 'cubic' /None (='nearest')
        self.signal_u = []
        self.signal_v = []
        self.signal_rgb = []
        for i in range(self.num_patch):
            self.rect.append(
                Rectangle((0, 0),
                          1,
                          1,
                          facecolor='None',
                          edgecolor=self.clrs[i]))
            self.points.append([])
            self.points_vis.append([])
            self.points_vis_obj.append([])
            if self.global_ref:
                self.points_ini.append([])
                self.points_vis_ini.append([])
        self.rect_drawn = 0
        self.rect_current = None
        self.rect_current_i = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.bboxes = [None] * self.num_patch
        # axes[0]
        flow = readFlow(flo_file)
        if crop is not None:
            flow = flow[crop[0]:crop[1], crop[2]:crop[3]]
        """if not self.org_video:
            flow_img, _, _ = flow_to_image(flow)
            self.img = self.ax.imshow(flow_img)
            self.ax.set_title('optical flow (color mapped)')"""
        #else:
        if first_frame is None:
            raise Exception('frame is not given (original video)')
        self.img = self.ax.imshow(first_frame)
        self.ax.set_title('original video')
        if self.plot_signal:
            flow_img, _, _ = flow_to_image(flow)
            self.img2 = self.ax_flo.imshow(flow_img)
            self.ax_flo.set_title('optical flow (color mapped)')
        for rect in self.rect:
            self.ax.add_patch(rect)
        #axes[1]
        if self.plot_signal:
            if self.record == 'displacement':
                self.ydata_u = np.zeros((150, self.num_patch))
                self.ydata_v = np.zeros((150, self.num_patch))
            else:
                self.ydata_v = np.zeros((150, 3))
                self.clrs = ['r', 'g', 'b']
            #self.ydata_u[-1] = flow_v
            self.graph_u, self.graph_v = [], []
            time_sec = np.array(range(150)) / 30
            if self.record == 'displacement':
                for i in range(self.num_patch):
                    graph, = self.ax_u.plot(time_sec,
                                            self.ydata_u[:, i],
                                            color=self.clrs[i])
                    self.graph_u.append(graph)
                    graph, = self.ax_v.plot(time_sec,
                                            self.ydata_v[:, i],
                                            color=self.clrs[i])
                    self.graph_v.append(graph)
            else:
                for i in range(3):
                    graph, = self.ax_v.plot(time_sec,
                                            self.ydata_v[:, i],
                                            color=self.clrs[i])
                    self.graph_v.append(graph)

            if self.record == 'displacement':
                self.ax_u.set_ylim([-1, 1])
                self.ax_u.set_title('mean flow (Mx)')
                self.ax_u.set_xlabel('Time (seconds)')
                self.ax_u.grid()
                self.ax_v.set_ylim([-1, 1])
                self.ax_v.set_title('mean flow (My)')
                self.ax_v.set_xlabel('Time (seconds)')
                self.ax_v.grid()
            else:
                self.ax_v.set_ylim([0, 255])
                self.ax_v.set_title('Mean intensity value (r,g,b)')
                self.ax_v.set_xlabel('Time (seconds)')
                self.ax_v.grid()

        # event
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.max_y, self.max_x = flow.shape[0], flow.shape[1]  # xy grid limit
        self.x_list = np.arange(0, self.max_x)
        self.y_list = np.arange(0, self.max_y)
        #x_grid, y_grid = np.meshgrid(x, y)  # it returns as dim(y)= 0, dim(x) = 1
        #self.xy_grid = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)
        self.fun_u = None
        self.fun_v = None
        # interp2D is 100x faster than griddata

    def restore_boxes(self, boxes):
        for i in range(self.num_patch):
            box = boxes[i]
            self.rect[i].set_width(box[1] - box[0])
            self.rect[i].set_height(box[3] - box[2])
            self.rect[i].set_xy((box[0], box[2]))
            self.rect[i].set_linestyle('solid')
            self.rect_current_i = i
            self.set_points(box)
        self.ax.figure.canvas.draw()
        self.bboxes = boxes
        self.rect_drawn = self.num_patch

    def set_points(self, bbox):
        if self.forced_size is None:
            x1, y1 = bbox[1], bbox[3]
        else:
            x1, y1 = bbox[0] + self.forced_size, bbox[2] + self.forced_size

        # points for calc grid points
        if self.calc_on:
            xs = np.arange(bbox[0], x1, self.points_res)
            ys = np.arange(bbox[2], y1, self.points_res)
            xgrid, ygrid = np.meshgrid(xs, ys)
            points = np.stack([xgrid.ravel(), ygrid.ravel()], axis=1)
            self.points[self.rect_current_i] = np.array(points,
                                                        dtype=np.float32)
            if self.global_ref:
                self.points_ini[self.rect_current_i] = np.array(
                    points, dtype=np.float32)

        # points for visualisation grid points
        if self.vis_on:
            points = []  # n*2 array
            points_obj = []
            for x in range(bbox[0], x1, self.points_vis_res):
                for y in range(bbox[2], y1, self.points_vis_res):
                    circle = Circle((x, y),
                                    self.points_vis_radius,
                                    color=self.clrs[self.rect_current_i])
                    self.ax.add_patch(circle)
                    points_obj.append(circle)
                    points.append([x, y])

            self.points_vis[self.rect_current_i] = np.array(points,
                                                            dtype=np.float32)
            if self.global_ref:
                self.points_vis_ini[self.rect_current_i] = np.array(
                    points, dtype=np.float32)
            for vis_obj in self.points_vis_obj[self.rect_current_i]:
                vis_obj.remove()
            self.points_vis_obj[self.rect_current_i] = points_obj

    def track_points(self,
                     points,
                     flow,
                     frame=None,
                     points_ini=None):  # only if self.on_img_grid=False
        if not self.interpolation:  # no  interpolation (use MV of nearest grid point)
            nearest_x = np.clip(points[:, 0].round().astype(np.int32), 0,
                                self.max_x - 1)
            nearest_y = np.clip(points[:, 1].round().astype(np.int32), 0,
                                self.max_y - 1)
            #points[:, 0] += flow[nearest_y, nearest_x, 0]  # x = x + u
            #points[:, 1] += flow[nearest_y, nearest_x, 1]  # y = y + v
            if self.global_ref:
                points = points_ini + flow[nearest_y, nearest_x]
            else:
                points += flow[nearest_y, nearest_x]
            if frame is None:  # Mx and My
                return np.mean(flow[nearest_y, nearest_x,
                                    0]), np.mean(flow[nearest_y, nearest_x, 1])
            else:  # R G B (face)
                return np.mean(frame[nearest_y, nearest_x, 0]), np.mean(frame[nearest_y, nearest_x, 1]),\
                       np.mean(frame[nearest_y, nearest_x, 2])
        else:
            """u_flow = griddata(self.xy_grid, flow[:, :, 0].ravel(), (points[:, 0], points[:, 1]),
                                     method=self.interpolation)
            v_flow = griddata(self.xy_grid, flow[:, :, 1].ravel(), (points[:, 0], points[:, 1]),
                                     method=self.interpolation)
            # x = x + u(flow)
            points[:, 0] += u_flow
            # y = y + v(flow)
            points[:, 1] += v_flow
            return np.mean(u_flow), np.mean(v_flow)"""
            if frame is not None:
                nearest_x = np.clip(points[:, 0].round().astype(np.int32), 0,
                                    self.max_x - 1)
                nearest_y = np.clip(points[:, 1].round().astype(np.int32), 0,
                                    self.max_y - 1)
            u_flow = np.zeros_like(points[:, 0])
            v_flow = np.zeros_like(points[:, 0])
            for i, point in enumerate(points):
                u_xy = self.fun_u([point[0]], [point[1]])
                v_xy = self.fun_v([point[0]], [point[1]])
                u_flow[i] = u_xy[0]
                v_flow[i] = v_xy[0]
            if self.global_ref:
                # x = x0 + u(flow)
                points[:, 0] = points_ini[:, 0] + u_flow
                # y = y0 + v(flow)
                points[:, 1] = points_ini[:, 1] + v_flow
            else:
                # x = x + u(flow)
                points[:, 0] += u_flow
                # y = y + v(flow)
                points[:, 1] += v_flow
            if frame is None:
                return np.mean(u_flow), np.mean(v_flow)
            else:
                return np.mean(frame[nearest_y, nearest_x, 0]), np.mean(frame[nearest_y, nearest_x, 1]),\
                       np.mean(frame[nearest_y, nearest_x, 2])

    def update_points(self, flow, frame=None):  # only if self.on_img_grid=False
        if self.interpolation is not None:
            self.fun_u = interp2d(self.x_list,
                                  self.y_list,
                                  flow[:, :, 0],
                                  kind=self.interpolation)
            self.fun_v = interp2d(self.x_list,
                                  self.y_list,
                                  flow[:, :, 1],
                                  kind=self.interpolation)
        if self.calc_on:
            data_u, data_v = [], []
            for i, points in enumerate(
                    self.points):  # for every patch (i.e. 6 patches)
                if frame is None:
                    if self.global_ref:
                        mean_u, mean_v = self.track_points(
                            points, flow, points_ini=self.points_ini[i])
                    else:
                        mean_u, mean_v = self.track_points(points, flow)
                    data_u.append(mean_u)
                    data_v.append(mean_v)
                else:  # will run only once within loop as num_patch = 2
                    if self.global_ref:
                        mean_r, mean_g, mean_b = self.track_points(
                            points, flow, frame, points_ini=self.points_ini[i])
                    else:
                        mean_r, mean_g, mean_b = self.track_points(
                            points, flow, frame)
                    self.signal_rgb.append([mean_r, mean_g, mean_b])
                if self.rect_move:
                    update_rect(points, self.rect[i])
            if frame is None:
                self.signal_u.append(data_u)
                self.signal_v.append(data_v)

        if self.vis_on:
            for points, points_ini, points_obj in zip(self.points_vis,
                                                      self.points_vis_ini,
                                                      self.points_vis_obj):
                #for i, (points, points_obj) in enumerate(zip(self.points_vis, self.points_vis_obj)):
                if self.global_ref:
                    self.track_points(points, flow, points_ini=points_ini)
                else:
                    self.track_points(points, flow)
                nearest_x = np.clip(points[:, 0].round(), 0, self.max_x - 1)
                nearest_y = np.clip(points[:, 1].round(), 0, self.max_y - 1)
                for i, circle in enumerate(points_obj):
                    circle.center = nearest_x[i], nearest_y[i]
                #if self.rect_move and not self.calc_on:
                #update_rect(points, self.rect[i])

    def update_data(self, flo_file, frame=None):  # frame is rgb
        flow = readFlow(flo_file)  # shape = (max_y, max_x, 2)

        if self.crop is not None:
            flow = flow[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
        if frame is not None:
            if flow.shape[0] != frame.shape[0] or flow.shape[1] != frame.shape[
                    1]:
                raise Exception(
                    "Image and flow file's x or y dimension doesn't match")
        """if not self.org_video:
            self.img.set_data(frame)
            if self.on_img_grid:
                flow_img, flow_u, flow_v = flow_to_image(flow, bboxes=self.bboxes)
                self.img.set_data(flow_img)
                self.signal_u.append(flow_u)
                self.signal_v.append(flow_v)
            else:
                flow_img, _, _ = flow_to_image(flow, bbox=self.bboxes[-1])  # focus on the last bbox (crop)
                self.img.set_data(flow_img)
                self.update_points(flow)"""
        #else:
        self.img.set_data(frame)
        if self.on_img_grid:
            flow_img, flow_u, flow_v = flow_to_image(flow, bboxes=self.bboxes)
            if self.record != 'displacement':
                rgb = img_to_mean_rgb(frame, bbox=self.bboxes[0])
                self.signal_rgb.append(rgb)
            else:
                self.signal_u.append(flow_u)
                self.signal_v.append(flow_v)
            if self.plot_signal:
                self.img2.set_data(flow_img)
        else:
            if self.record != 'displacement':
                self.update_points(flow, frame=frame)
            else:
                self.update_points(flow)
            if self.plot_signal:
                flow_img, _, _ = flow_to_image(flow,
                                               bbox=self.bboxes[-1],
                                               const_rad_max=None,
                                               filter_upper_range=False)
                self.img2.set_data(flow_img)

        if self.plot_signal:
            if self.record != 'displacement':
                self.ydata_v[:-1] = self.ydata_v[1:]
                self.ydata_v[-1] = self.signal_rgb[-1]
                for i, graph in enumerate(self.graph_v):
                    graph.set_ydata(self.ydata_v[:, i])
            else:
                self.ydata_u[:-1] = self.ydata_u[1:]
                self.ydata_u[-1] = self.signal_u[-1]
                for i, graph in enumerate(self.graph_u):
                    graph.set_ydata(self.ydata_u[:, i])

                self.ydata_v[:-1] = self.ydata_v[1:]
                self.ydata_v[-1] = self.signal_v[-1]
                for i, graph in enumerate(self.graph_v):
                    graph.set_ydata(self.ydata_v[:, i])

    def on_press(self, event):
        #print('press')
        self.x0 = event.xdata
        self.y0 = event.ydata
        if not self.x0:
            return
        if self.rect_drawn < self.num_patch:  # not all the patches drawn yet
            self.rect_current = self.rect[self.rect_drawn]
            self.rect_current_i = self.rect_drawn
        else:
            np_bbox = np.array(self.bboxes)
            diff = np.abs(np_bbox[:, 0] - self.x0) + np.abs(np_bbox[:, 2] -
                                                            self.y0)
            self.rect_current_i = np.argmin(diff)
            self.rect_current = self.rect[self.rect_current_i]
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect_current.set_width(self.x1 - self.x0)
        self.rect_current.set_height(self.y1 - self.y0)
        self.rect_current.set_xy((self.x0, self.y0))
        self.rect_current.set_linestyle('dashed')
        self.ax.figure.canvas.draw()
        self.pressed = True

    def on_motion(self, event):
        if not self.pressed:
            return
        self.x1 = event.xdata
        self.y1 = event.ydata
        if not self.x1:
            return
        self.rect_current.set_width(self.x1 - self.x0)
        self.rect_current.set_height(self.y1 - self.y0)
        self.rect_current.set_xy((self.x0, self.y0))
        self.rect_current.set_linestyle('dashed')
        self.ax.figure.canvas.draw()

    def on_release(self, event):
        #print('release')
        self.x1 = event.xdata
        self.y1 = event.ydata
        if not self.x1:
            return
        if not self.forced_size:  # free draw for last box
            self.rect_current.set_width(self.x1 - self.x0)
            self.rect_current.set_height(self.y1 - self.y0)
        else:
            self.rect_current.set_width(self.forced_size)
            self.rect_current.set_height(self.forced_size)
            self.x1, self.y1 = self.x0 + self.forced_size, self.y0 + self.forced_size
        self.rect_current.set_xy((self.x0, self.y0))
        self.rect_current.set_linestyle('solid')
        self.ax.figure.canvas.draw()
        self.pressed = False
        if self.rect_drawn < self.num_patch:
            self.rect_drawn = self.rect_drawn + 1
        print(floor(self.x0), floor(self.x1), floor(self.y0), floor(self.y1))
        bbox = [floor(self.x0), floor(self.x1), floor(self.y0), floor(self.y1)]
        self.bboxes[self.rect_current_i] = bbox

        self.set_points(bbox)
        #return [self.x0, self.x1, self.y0, self.y1]


def laod_box_info(set_id, vid, record):
    box_info = {}
    filename = 'outputs/' + record + '_' + vid + '.json'
    if os.path.exists(filename):
        with open(filename, 'r') as fp:
            box_info = json.load(fp)
    if str(set_id) in box_info.keys():
        if input(
                "replace previously drawn boxes for current set? (y/n)") == 'y':
            return box_info, None
        else:
            print("Restoring previously drawn boxes .. ")
            return box_info, box_info[str(set_id)]["boxes"]
    else:
        return box_info, None


def collect_signals(vid,
                    vid_path,
                    flo_path,
                    record,
                    apply_mv=True,
                    square_size=None,
                    interp=None,
                    num_square=1,
                    vis_on=True,
                    calc_on=True,
                    vis_res=5,
                    calc_res=1,
                    radius=2,
                    start_sec=0,
                    end_sec=None,
                    graph_on=False,
                    rect_move=False,
                    crop=None,
                    global_ref=True,
                    set_id=0,
                    fx=1.0,
                    fy=1.0,
                    output_folder='outputs/'):
    img_grid = False if apply_mv else True

    boxes_info, current_box_set = laod_box_info(set_id, vid, record)
    print(current_box_set)
    boxes = current_box_set
    if boxes is not None:
        num_square = len(boxes)
    matplotObj = None
    img = None
    #if record == 'rgb':
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_at = int(start_sec * fps)

    if end_sec is None:
        end_at = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) -
                     1)  # flow files are 1 less than num of frames
    else:
        end_at = int(end_sec * fps)
    assert 1 + start_at < cap.get(
        cv2.CAP_PROP_FRAME_COUNT), "invalid start second given"
    assert 1 + end_at <= cap.get(
        cv2.CAP_PROP_FRAME_COUNT), "invalid end second given"

    cap.set(cv2.CAP_PROP_POS_FRAMES,
            1 + start_at)  # we get flow file from 2nsd frame

    for flo in sorted(os.listdir(flo_path))[start_at:end_at]:
        #if record == 'rgb':
        ret, img = cap.read()
        if not ret:
            print('video reading error')
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if crop is not None:
            img = img[crop[0]:crop[1], crop[2]:crop[3]]
        img = cv2.resize(img, (0, 0), fx=fx, fy=fy)
        img = cv2.rectangle(img, (20, 0), (200, 200), (0, 0, 0), -1) # hide face for 'face002'
        #img = cv2.rectangle(img, (500, 0), (700, 200), (0, 0, 0), -1) # hide face for 'dog006'

        if crop is not None:
            img = img[crop[0]:crop[1], crop[2]:crop[3]]
            #print(img.shape)
        if not matplotObj:
            matplotObj = Annotate(os.path.join(flo_path, flo),
                                  img_grid,
                                  forced_size=square_size,
                                  interp=interp,
                                  num_patch=num_square,
                                  vis_on=vis_on,
                                  calc_on=calc_on,
                                  vis_res=vis_res,
                                  calc_res=calc_res,
                                  vis_radius=radius,
                                  first_frame=img,
                                  plot_signal=graph_on,
                                  rect_move=rect_move,
                                  crop=None,
                                  record=record,
                                  global_ref=global_ref)
            if boxes is not None:
                matplotObj.restore_boxes(boxes)
                plt.pause(0.001)
            else:
                print(
                    'Draw the rectangular box on the figure labelled as original video'
                )
                print(
                    'This will be the region from where you want to collect signal'
                )
                print(
                    'To draw drag the mouse cursor from left top and end bottom, not the opposite'
                )
            print(
                'press q to stop the process.. you can still save the unfinished result'
            )
        else:
            matplotObj.update_data(os.path.join(flo_path, flo), frame=img)
        plt.draw()

        if matplotObj.rect_drawn < matplotObj.num_patch:
            while matplotObj.rect_drawn < matplotObj.num_patch:
                plt.pause(1)
            matplotObj.update_data(os.path.join(flo_path, flo), frame=img)
        mypause(0.001)
        if not plt.get_fignums():  # 'press q to exit (handled by matplotlib)'
            break
    plt.close()

    if input("store the signal values ?") == 'y':
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        #if boxes is None:
        info = {}
        info["boxes"] = matplotObj.bboxes
        info["start_frame(first)"] = start_at
        info["end_frame(first)"] = end_at
        info["fps"] = fps
        info["crop"] = crop
        info["fx"] = fx
        info["fy"] = fy
        info["vid_id"] = vid
        info["vid_path"] = vid_path
        info["flow_folder"] = flo_path
        boxes_info[str(set_id)] = info

        if record == 'displacement':
            data_u = np.array(matplotObj.signal_u, dtype=np.float32)
            data_v = np.array(matplotObj.signal_v, dtype=np.float32)
            data = np.concatenate((data_u, data_v), axis=1)
            np.save(output_folder + 'displacement_' + vid + '_' + str(set_id) + '.npy',
                    data)
            with open(output_folder + 'displacement_' + vid + '.json', 'w') as fp:
                json.dump(boxes_info, fp)
            print('saved to '+output_folder + 'displacement_' + vid + '_' + str(set_id) + '.npy')
        else:
            data = np.array(matplotObj.signal_rgb, dtype=np.float32)
            np.save(output_folder + 'rgb_' + vid + '_' + str(set_id) + '.npy',
                    data)
            with open(output_folder + 'rgb_' + vid + '.json', 'w') as fp:
                json.dump(boxes_info, fp)


if __name__ == "__main__":
    vid_path = '/home/mrahman7/Documents/ratVideos/rat001.MOV'
    flo_path = '/home/mrahman7/Documents/of_out/rat001'
    collect_signals('rat001',
                    vid_path,
                    flo_path,
                    'rgb',
                    apply_mv=True,
                    square_size=None,
                    num_square=1,
                    vis_on=True,
                    calc_on=True,
                    vis_res=10,
                    calc_res=1,
                    radius=2,
                    graph_on=True,
                    rect_move=False,
                    crop=None,
                    start_sec=0,
                    end_sec=3,
                    global_ref=True,
                    set_id=3,
                    fx=0.25,
                    fy=0.25,
                    output_folder='outputs/')
    #collect_signals('face004', False, interp='cubic', square_size=None, num_square=2, vis_on=True, calc_on=True,
    #vis_res=10, calc_res=1, radius=2, org_video=True, graph_on=True, rect_move=False,
    #crop=None, start_at=0, end_at=-1, record='pixel', global_ref=True, set_id=0)
