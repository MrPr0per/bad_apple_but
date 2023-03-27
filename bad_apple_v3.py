import numpy as np
import numba as nb
import cv2

vidcap = cv2.VideoCapture('bad_apple_3.mp4')
FPS = vidcap.get(cv2.CAP_PROP_FPS) + 2

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = vidcap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('rec_bad_apple10.mp4', fourcc, fps, (width, height))

success, image = vidcap.read()

n_tiles = 24
im_h, im_w = image.shape[:2]  # ширина и высота картинки
tile_w = int(im_w / n_tiles)
tile_h = int(im_h / n_tiles)


@nb.njit(fastmath=True)
def change_color(color, local_br, global_br):
    # нужно сделать количество белого равным local_br, но не переходя границ offset
    offset = 0.12  # граничное значение белого
    r, g, b = color
    if offset < global_br < 1 - offset:
        final_value = max(offset, min(1 - offset, local_br))  # нужно домножить цвет так, чт. global стал final

        if global_br > 0.5:
            if final_value > 0.5:
                if final_value > global_br:
                    # инвертировать цвета
                    r, g, b = 255 - r, 255 - g, 255 - b
                    final_value = 1 - final_value
                    # домножить
                    k = final_value / global_br
                    r, g, b = r * k, g * k, b * k
                    # инвертировать обратно
                    r, g, b = 255 - r, 255 - g, 255 - b
                    final_value = 1 - final_value
                else:
                    # домножить
                    k = final_value / global_br
                    r, g, b = r * k, g * k, b * k
            else:
                if 1 - global_br > final_value:
                    # инвертировать цвета
                    r, g, b = 255 - r, 255 - g, 255 - b
                    # домножить
                    k = final_value / global_br
                    r, g, b = r * k, g * k, b * k
                else:
                    # домножить
                    k = final_value / global_br
                    r, g, b = r * k, g * k, b * k
        else:
            if final_value > 0.5:
                if 1 - final_value < global_br:
                    # инвертировать цель
                    final_value = 1 - final_value
                    # домножить
                    k = final_value / global_br
                    r, g, b = r * k, g * k, b * k
                    # инветировать все обратно
                    r, g, b = 255 - r, 255 - g, 255 - b
                    final_value = 1 - final_value
                else:
                    # инвертировать цвета
                    r, g, b = 255 - r, 255 - g, 255 - b
                    # домножить
                    k = final_value / global_br
                    r, g, b = r * k, g * k, b * k
            else:
                if final_value < global_br:
                    # домножить
                    k = final_value / global_br
                    r, g, b = r * k, g * k, b * k
                else:
                    # инвертировать цвета
                    r, g, b = 255 - r, 255 - g, 255 - b
                    # домножить
                    k = final_value / global_br
                    r, g, b = r * k, g * k, b * k

    else:
        if (local_br < 0.5 and global_br > 0.5) or (local_br > 0.5 and global_br < 0.5):
            r, g, b = 255 - r, 255 - g, 255 - b

    return r, g, b


@nb.njit(fastmath=True)
def gen_new_image(image, n_tiles, tile_w, tile_h):
    im_h, im_w = image.shape[:2]  # ширина и высота картинки
    outp_image = np.zeros((im_h, im_w, 3), dtype=np.uint8)

    # x, y      координаты текущей плитки
    # dx, dy    относитеотные координаты пикселя в текущей плитке
    # rx, ry    реальные координаты текущего пикселя
    # nx, ny    новые реальные координаты текущего пикселя
    global_brightness = 0
    count = 0
    for y in range(0, im_h, 10):
        for x in range(0, im_w, 10):
            global_brightness += image[y][x][0]
            count += 1
    global_brightness = global_brightness / count / 255
    for y in range(0, n_tiles):
        for x in range(0, n_tiles):
            # определение яркости
            local_brightness = 0
            count = 0
            for dy in range(tile_h):
                for dx in range(tile_w):
                    ry = y * tile_h + dy
                    rx = x * tile_w + dx
                    local_brightness += image[ry][rx][0]
                    count += 1
            local_brightness = local_brightness / count / 255

            for dy in range(tile_h):
                for dx in range(tile_w):
                    nx = x + dx * n_tiles
                    ny = y + dy * n_tiles
                    ry = y * tile_h + dy
                    rx = x * tile_w + dx

                    color = image[ny][nx]
                    new_color = change_color(color, local_brightness, global_brightness)
                    outp_image[ry][rx] = new_color
    return outp_image


while True:
    success, image = vidcap.read()
    if not success: break
    outp_image = gen_new_image(image, n_tiles, tile_w, tile_h)
    cv2.imshow('video', outp_image)
    out.write(outp_image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

vidcap.release()
out.release()
cv2.destroyAllWindows()
