import argparse
import numpy as np
from utils.visual import get_pixel_coord_from_world
from utils.visual import action_viz
import pyflex
import os, random
from tqdm import tqdm
import imageio
import pickle
from softgym.envs.foldenv import FoldEnv

def rotate_vector(vector, angle, direction='clockwise'):
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Define rotation matrices
    if direction == 'clockwise':
        rotation_matrix = np.array([
            [np.cos(angle_rad), np.sin(angle_rad)],
            [-np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif direction == 'counterclockwise':
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
    else:
        raise ValueError("Invalid direction: choose 'clockwise' or 'counterclockwise'")
    
    # Extract the x and z components
    x, z = vector[0], vector[2]
    xz_vector = np.array([x, z])
    
    # Rotate the vector
    rotated_xz = np.dot(rotation_matrix, xz_vector)
    
    # Return the rotated vector with the middle element unchanged
    return np.array([rotated_xz[0], vector[1], rotated_xz[1]])

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    # Oracle demonstration
    parser = argparse.ArgumentParser(description="Generate Demonstrations")
    parser.add_argument("--gui", action="store_true", help="Run headless or not")
    parser.add_argument("--corner_bias", action="store_true", help="Task name")
    parser.add_argument("--img_size", type=int, default=224, help="Size of rendered image")
    parser.add_argument("--cached", type=str, help="Cached filename")
    parser.add_argument("--horizon", type=int, default=1, help="Number of horizons in a episode")
    parser.add_argument("--attempts", type=int, default=50, help="Number of random action attempts for a particular config")
    parser.add_argument("--short_horizon", action="store_true", help="Run script for short horizon actions")
    parser.add_argument("--inverse_dynamics", action="store_true", help="Run script for generating Inverse Dynamics questions")
    parser.add_argument("--changing_angles", action="store_true", help="Either change the angles for Inverse Dynamics or change distances")
    args = parser.parse_args()

    # env settings
    cached_path = os.path.join("cached configs", args.cached + ".pkl")
    env = FoldEnv(cached_path, gui=args.gui, render_dim=args.img_size)

    # save settings
    horizon_path = "short horizon" if args.short_horizon else "long horizon"
    dynamics_path = "inverse dynamics" if args.inverse_dynamics else "forward dynamics"
    angles_path = "changing angles" if args.changing_angles else "changing distances"
    save_path = os.path.join("data", "random", "corner bias" if args.corner_bias else "random", str(args.horizon), horizon_path, dynamics_path, angles_path)
    os.makedirs(save_path, exist_ok=True)

    # other settings
    rgb_shape = (args.img_size, args.img_size)
    num_data = env.num_configs

    max_index = -1
    # dirs = os.listdir(save_path)
    # if dirs == []:
    #     max_index = -1
    # else:
    #     existed_index = np.array(dirs).astype(int)
    #     max_index = existed_index.max()

    for config_id in tqdm(range(49, min(num_data, 400))):
        # folders
        save_folder_root = os.path.join(save_path, str(config_id + max_index + 1))

        for attempt in range(args.attempts):
            save_folder = os.path.join(save_folder_root, str(attempt))
            save_folder_rgb = os.path.join(save_folder, "rgb")
            save_folder_depth = os.path.join(save_folder, "depth")
            os.makedirs(save_folder, exist_ok=True)
            os.makedirs(save_folder_rgb, exist_ok=True)
            os.makedirs(save_folder_depth, exist_ok=True)

            pick_pixels = []
            place_pixels = []
            rgbs = []

            # env reset
            env.reset(config_id=config_id)
            camera_params = env.camera_params
            rgb, depth = env.render_image()
            imageio.imwrite(os.path.join(save_folder_rgb, str(0) + ".png"), rgb)
            depth = depth * 255
            depth = depth.astype(np.uint8)
            imageio.imwrite(os.path.join(save_folder_depth, str(0) + ".png"), depth)
            rgbs.append(rgb)

            center = np.zeros(3)
            if args.corner_bias:
                if args.inverse_dynamics:
                    # action viz
                    save_folder_viz = os.path.join(save_folder, "rgbviz")
                    os.makedirs(save_folder_viz, exist_ok=True)

                    # set direction
                    corners = env.get_corners()
                    corner_index = np.random.randint(0, 4)
                    pick_pos = corners[corner_index]

                    diff = center - pick_pos
                    direction = np.where(diff >= 0, 1, -1)
                    random_low = 0.07 if args.short_horizon else 0.05
                    random_high = 0.12 if args.short_horizon else 0.3
                    range_size = random_high - random_low

                    # Getting the correct action first
                    correct_action = np.random.uniform(random_low, random_high, (3,))
                    correct_action = correct_action * direction
                    correct_action[1] = 0

                    # Rotating the correct action for the case of a short horizon action
                    if args.short_horizon and args.changing_angles:
                        angle = random.uniform(0, 90)
                        correct_action = rotate_vector(correct_action, angle)

                    place_pos = pick_pos + correct_action
                    place_pos = np.clip(place_pos, -0.4, 0.4)
                    env.pick_and_place(pick_pos.copy(), place_pos.copy())

                    rgb, depth = env.render_image()
                    pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                    place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)

                    # save the state after the action
                    pick_pixels.append(pick_pixel)
                    place_pixels.append(place_pixel)
                    depth = depth * 255
                    depth = depth.astype(np.uint8)
                    imageio.imwrite(os.path.join(save_folder_rgb, str(1) + ".png"), rgb)
                    imageio.imwrite(os.path.join(save_folder_depth, str(1) + ".png"), depth)
                    rgbs.append(rgb)

                    # env reset
                    env.reset(config_id=config_id)
                    camera_params = env.camera_params
                    rgb, depth = env.render_image()

                    # writing the correct action first -> Always in index: 0
                    img = action_viz(rgb, pick_pixel, place_pixel)
                    imageio.imwrite(os.path.join(save_folder_viz, "0" + ".png"), img)

                    # writing the options now. We choose three incorrect options
                    action_list = [correct_action]
                    option_index = 1
                    tolerance = 1.25 if args.short_horizon else 0.35
                    if args.short_horizon and args.changing_angles:
                        tolerance = 0.89
                    while option_index < 4:
                        # env reset
                        env.reset(config_id=config_id)
                        camera_params = env.camera_params
                        rgb, depth = env.render_image()

                        # Getting a random action
                        random_action = np.random.uniform(random_low, random_high, (3,))
                        random_action = random_action * direction

                        # Getting changing position for short horizon case
                        if args.short_horizon:
                            random_action = random_action * np.random.uniform(2.0, 2.35)
                        random_action[1] = 0

                        if args.changing_angles:
                            direction_list = ["clockwise", "counterclockwise"]
                            random_angle = random.uniform(40, 80)
                            random_direction = random.choice(direction_list)
                            random_action = rotate_vector(correct_action, random_angle, random_direction)
                        random_action[1] = 0

                        # Check if the random action is different from the correct action
                        use_random = True
                        for action in action_list:
                            dist = np.linalg.norm(random_action - action)
                            if dist <= tolerance * range_size:
                                use_random = False
                                break
                        
                        if not use_random:
                            continue

                        place_pos = pick_pos + random_action
                        place_pos = np.clip(place_pos, -0.4, 0.4)
                        
                        # Getting the pick and place pixels to get the RGB visualizations
                        pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                        place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)
                        img = action_viz(rgb, pick_pixel, place_pixel)
                        imageio.imwrite(os.path.join(save_folder_viz, str(option_index) + ".png"), img)
                        option_index += 1
                        action_list.append(random_action)
                else:
                    # corner bias
                    for i in range(args.horizon):
                        # set direction
                        corners = env.get_corners()
                        pick_pos = corners[np.random.randint(0, 4)]

                        diff = center - pick_pos
                        direction = np.where(diff >= 0, 1, -1)

                        random_low = 0.05 if args.short_horizon else 0.05
                        random_high = 0.15 if args.short_horizon else 0.3
                        random_action = np.random.uniform(random_low, random_high, (3,))
                        random_action = random_action * direction
                        random_action[1] = 0

                        place_pos = pick_pos + random_action
                        place_pos = np.clip(place_pos, -0.4, 0.4)
                        env.pick_and_place(pick_pos.copy(), place_pos.copy())

                        rgb, depth = env.render_image()
                        pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                        place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)

                        # save
                        pick_pixels.append(pick_pixel)
                        place_pixels.append(place_pixel)
                        depth = depth * 255
                        depth = depth.astype(np.uint8)
                        imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                        imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)
                        rgbs.append(rgb)

            else:
                # random action
                for i in range(args.horizon):
                    particle_pos = env.action_tool._get_pos()[1]
                    pick_pos = particle_pos[np.random.randint(particle_pos.shape[0])][0:3]

                    diff = center - pick_pos
                    direction = np.where(diff >= 0, 1, -1)
                    random_action = np.random.uniform(0.05, 0.2, (3,))
                    random_action = random_action * direction
                    random_action[1] = 0

                    place_pos = pick_pos + random_action
                    place_pos = np.clip(place_pos, -0.4, 0.4)
                    env.pick_and_place(pick_pos.copy(), place_pos.copy())

                    rgb, depth = env.render_image()
                    pick_pixel = get_pixel_coord_from_world(pick_pos, rgb_shape, camera_params)
                    place_pixel = get_pixel_coord_from_world(place_pos, rgb_shape, camera_params)

                    # save
                    pick_pixels.append(pick_pixel)
                    place_pixels.append(place_pixel)
                    depth = depth * 255
                    depth = depth.astype(np.uint8)
                    imageio.imwrite(os.path.join(save_folder_rgb, str(i + 1) + ".png"), rgb)
                    imageio.imwrite(os.path.join(save_folder_depth, str(i + 1) + ".png"), depth)
                    rgbs.append(rgb)

            with open(os.path.join(save_folder, "info.pkl"), "wb+") as f:
                data = {"pick": pick_pixels, "place": place_pixels}
                pickle.dump(data, f)

            if not args.inverse_dynamics:
                # action viz
                save_folder_viz = os.path.join(save_folder, "rgbviz")
                os.makedirs(save_folder_viz, exist_ok=True)

                num_actions = len(pick_pixels)

                for i in range(args.horizon+1):
                    if i < num_actions:
                        img = action_viz(rgbs[i], pick_pixels[i], place_pixels[i])
                    else:
                        img = rgbs[i]
                    imageio.imwrite(os.path.join(save_folder_viz, str(i) + ".png"), img)

if __name__ == "__main__":
    main()
