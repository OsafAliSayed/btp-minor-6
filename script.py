from PIL import Image, ImageDraw

from PIL import Image, ImageDraw

from PIL import Image, ImageDraw

def binary_visualization(input_text, output_image_text):
    # Convert each character to its ASCII value
    ascii_values = [ord(char) for char in input_text]

    # Calculate the size of the image based on the string length
    width = int(len(ascii_values)**0.5)
    height = (len(ascii_values) // width) + 1

    # Create a new image with white background
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    shade_range = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    for color in shade_range:
        # Draw pixels based on the ASCII values
        for y in range(height):
            for x in range(width):
                index = y * width + x
                if index < len(ascii_values):
                    # Map ASCII value to a shade of grey
                    shade = ascii_values[index] * color // 127
                    draw.point((x, y), fill=(shade, shade, shade))
        img.save(output_image_path + str(color) + ".png")
    return img
# def text_to_binary_image(input_text, output_image_path):
#     # Convert each character in the input text to binary representation
#     binary_string = ''.join(format(ord(char), '08b') for char in input_text)

#     # Calculate the size of the image based on the binary string length
#     width = int(len(binary_string)**0.5)
#     height = (len(binary_string) // width) + 1

#     # Create a new image with white background
#     img = Image.new('RGB', (width, height), 'white')
#     draw = ImageDraw.Draw(img)

#     # Draw black pixels based on the binary string
#     for y in range(height):
#         for x in range(width):
#             index = y * width + x
#             if index < len(binary_string) and binary_string[index] == '1':
#                 draw.point((x, y), fill='black')

#     # Save the image
#     img.save(output_image_path)

if __name__ == "__main__":
    input_text = input("Enter the text to convert to binary image: ")
    output_image_path = input("Enter the output image path (without extension, e.g., output): ")

    binary_visualization(input_text, output_image_path)
    print(f"Binary image saved to {output_image_path}")
