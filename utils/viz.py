import pdb
from visdom import Visdom
import numpy as np

class SingleVisdom:
    __instance = {}
    @staticmethod
    def getInstance(env="main"):
        if env not in SingleVisdom.__instance:
            SingleVisdom(env)
        return SingleVisdom.__instance[env]
    
    def __init__(self, env="main"):
        if env in SingleVisdom.__instance:
            raise Exception("Singleton constructor called twice for environment: {}!".format(env))
        else:    
            SingleVisdom.__instance[env] = Visdom(env=env)

    @staticmethod
    def reset_window():
        viz = SingleVisdom.getInstance()
        for env in viz.get_env_list():
            env_viz = SingleVisdom.getInstance(env=env)
            env_viz.close(win=None)
        print("Finished resetting Visdom environments")


class VisdomPlotter:
    def __init__(self, env="main"):
        self.viz = SingleVisdom.getInstance(env=env)

    
"""
Plots multiple graphs in a given environment
"""
class VisdomLinePlotter(VisdomPlotter):
    def __init__(self, env="main"):
        super().__init__(env=env)
        self.env = env
        self.plots = {}
    
    def register_plot(self, name, opts=None):
        self.plots[name] = {}
        self.plots[name]["name"] = name
        self.plots[name]["opts"] = opts

    # x, y can either be list or scalars values
    def plot(self, name, split_name, x, y):
        if name not in self.plots:
            raise Exception("No plot registered with name: {}".format(name))
        plot_obj = self.plots[name]

        # Update legend
        opts = plot_obj["opts"]        
        if "legend" not in opts:
            opts["legend"] = []
        if split_name not in opts["legend"]:
            opts["legend"].append(split_name)

        if type(x) is list:
            X = np.array(x)
            Y = np.array(y)
        else:
            X = np.array([x])
            Y = np.array([y])

        if "plot" not in plot_obj:
            # initialize plot with first point
            plot_obj["plot"] = self.viz.line(X=X,
                Y=Y,
                env=self.env,
                name=split_name,
                opts=opts)
        else:
            # append new point to plot
            self.viz.line(X=X,
                Y=Y,
                env=self.env,
                win=plot_obj["plot"],
                name=split_name,
                update = "append")

"""
Plots training and validation losses
"""
class LossPlotter(VisdomLinePlotter):
    def __init__(self, plot_name, title, len_loader, env="main", epoch=0):
        super().__init__(env=env)
        self.train_epoch = epoch    # may be fractional values
        self.val_epoch = epoch
        self.plot_name = plot_name
        self.len_loader = len_loader
        self.register_plot(plot_name, opts=dict(
            title=title,
            xlabel="epoch",
            ylabel="loss",
        ))

        self.data = {} # data.split_name.X and data.split_name.Y

    def export_state(self):
        state = dict(
            data = self.data,
            train_epoch = self.train_epoch,
            val_epoch = self.val_epoch,
            plot_name = self.plot_name,
            len_loader = self.len_loader
        )
        return state

    def load_from_state(self, state):
        self.train_epoch = state["train_epoch"]
        self.val_epoch = state["val_epoch"]
        self.plot_name = state["plot_name"]
        self.len_loader = state["len_loader"]
        self.data = state["data"]

        for split_name, data in state["data"].items():
            self.plot(self.plot_name, split_name, data["X"], data["Y"])

        print("Loaded visdom from state")

    def increment_train_epoch(self):
        self.train_epoch += 1 / self.len_loader
    
    def increment_val_epoch(self):
        self.val_epoch += 1

    def update_state(self, split_name, epoch, loss):
        if split_name not in self.data:
            self.data[split_name] = {
                "X": [],
                "Y": []
            }        
        self.data[split_name]["X"].append(epoch)
        self.data[split_name]["Y"].append(loss)

    def plot_train(self, loss): 
        self.update_state("train", self.train_epoch, loss)     
        self.plot(self.plot_name, "train", self.train_epoch, loss)

    def plot_val(self, loss):
        self.update_state("val", self.val_epoch, loss)
        self.plot(self.plot_name, "val", self.val_epoch, loss)


class PlotlyPlotter(VisdomPlotter):
    def __init__(self, env="main"):
        super().__init__(env=env)
        self.env = env

    def plot(self, fig):
        self.viz.plotlyplot(fig)    

class ImagePlotter(VisdomPlotter):
    def __init__(self, env="main"):
        super().__init__(env=env)
        self.env = env

    # jpgquality: 
    #   JPG quality (number 0-100). 
    #   If defined image will be saved as JPG to reduce file size. 
    #   If not defined image will be saved as PNG.
    # caption: 
    #   Caption for the image
    # store_history: 
    #   Keep all images stored to the same window and attach a slider
    #   to the bottom that will let you select the image to view. 
    #   You must always provide this opt when sending new images to an image with history.
    def plot_image(self, image, win=None, opts=None):
        self.viz.image(image, win=win, opts=opts)



    # nrow: Number of images in a row
    # padding: Padding around the image, equal padding around all 4 sides
    # opts.jpgquality: 
    #   JPG quality (number 0-100). 
    #   If defined image will be saved as JPG to reduce file size. 
    #   If not defined image will be saved as PNG.
    # opts.caption: Caption for the image
    def plot_images(self, images, opts=None):
        self.viz.images(images, opts=opts)
