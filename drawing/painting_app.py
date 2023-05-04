import numpy as np

from random import random
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Line
from kivy.graphics.texture import Texture
from PIL import Image
from io import BytesIO


class PaintWidget(Widget):
    def on_touch_down(self, touch):
        color = (random(), 1, 1)
        with self.canvas:
            Color(255, 255, 255)
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=12)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class PaintApp(App):

    def build(self):
        parent = Widget()
        self.painter = PaintWidget()

        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release=self.clear_canvas)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        return parent

    def clear_canvas(self, obj):
        self.painter.canvas.clear()
        self.get_pixels_vector()

    def get_pixels_vector(self):
        img = self.painter.export_as_image(keep_data=True)
        print(img)


        pixels_vector = np.zeros((self.painter.size[0] * self.painter.size[1], 1))
        # for i in range(self.painter.size[0]):
        #     for j in range(self.painter.size[1]):
        #         pixels_vector[0][i+j] = img.read_pixel(i, j)




if __name__ == '__main__':
    PaintApp().run()
