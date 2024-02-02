import argparse
import asyncio
import contextlib
import json
import os
import random
import sys
import time
from functools import partial
from typing import Dict
from typing import List
from typing import Union
from slugify import slugify

import httpx
import pkg_resources
import regex
import requests

BING_URL = os.getenv("BING_URL", "https://www.bing.com")
# Generate random IP between range 13.104.0.0/14
FORWARDED_IP = (
    f"13.{random.randint(104, 107)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
)
HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "max-age=0",
    "content-type": "application/x-www-form-urlencoded",
    "referrer": "https://www.bing.com/images/create/",
    "origin": "https://www.bing.com",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.63",
    "x-forwarded-for": FORWARDED_IP,
}

# Error messages
error_timeout = "Your request has timed out."
error_redirect = "Redirect failed"
error_blocked_prompt = (
    "Your prompt has been blocked by Bing. Try to change any bad words and try again."
)
error_being_reviewed_prompt = "Your prompt is being reviewed by Bing. Try to change any sensitive words and try again."
error_noresults = "Could not get results"
error_unsupported_lang = "\nthis language is currently not supported by bing"
error_bad_images = "Bad images"
error_no_images = "No images"
# Action messages
sending_message = "Sending request..."
wait_message = "Waiting for results..."
download_message = "\nDownloading images..."


def debug(debug_file, text_var):
    """helper function for debug"""
    with open(f"{debug_file}", "a", encoding="utf-8") as f:
        f.write(str(text_var))
        f.write("\n")


class ImageGen:
    """
    Image generation by Microsoft Bing
    Parameters:
        auth_cookie: str
        auth_cookie_SRCHHPGUSR: str
    Optional Parameters:
        debug_file: str
        quiet: bool
        all_cookies: List[Dict]
    """

    def __init__(
            self,
            auth_cookie: str,
            auth_cookie_SRCHHPGUSR: str,
            debug_file: Union[str, None] = None,
            quiet: bool = False,
            all_cookies: List[Dict] = None,
    ) -> None:
        self.session: requests.Session = requests.Session()
        self.session.headers = HEADERS
        self.session.cookies.set("_U", auth_cookie)
        self.session.cookies.set("SRCHHPGUSR", auth_cookie_SRCHHPGUSR)
        if all_cookies:
            for cookie in all_cookies:
                self.session.cookies.set(cookie["name"], cookie["value"])
        self.quiet = quiet
        self.debug_file = debug_file
        if self.debug_file:
            self.debug = partial(debug, self.debug_file)

    def get_images(self, prompt: str) -> list:
        """
        Fetches image links from Bing
        Parameters:
            prompt: str
        """
        if not self.quiet:
            print(sending_message)
        if self.debug_file:
            self.debug(sending_message)
        url_encoded_prompt = requests.utils.quote(prompt)
        payload = f"q={url_encoded_prompt}&qs=ds"
        # https://www.bing.com/images/create?q=<PROMPT>&rt=3&FORM=GENCRE
        url = f"{BING_URL}/images/create?q={url_encoded_prompt}&rt=4&FORM=GENCRE"
        response = self.session.post(
            url,
            allow_redirects=False,
            data=payload,
            timeout=200,
        )
        # check for content waring message
        if "this prompt is being reviewed" in response.text.lower():
            if self.debug_file:
                self.debug(f"ERROR: {error_being_reviewed_prompt}")
            raise Exception(
                error_being_reviewed_prompt,
            )
        if "this prompt has been blocked" in response.text.lower():
            if self.debug_file:
                self.debug(f"ERROR: {error_blocked_prompt}")
            raise Exception(
                error_blocked_prompt,
            )
        if (
                "we're working hard to offer image creator in more languages"
                in response.text.lower()
        ):
            if self.debug_file:
                self.debug(f"ERROR: {error_unsupported_lang}")
            raise Exception(error_unsupported_lang)
        if response.status_code != 302:
            # if rt4 fails, try rt3
            url = f"{BING_URL}/images/create?q={url_encoded_prompt}&rt=3&FORM=GENCRE"
            response = self.session.post(url, allow_redirects=False, timeout=200)
            if response.status_code != 302:
                if self.debug_file:
                    self.debug(f"ERROR: {error_redirect}")
                print(f"ERROR: {response.text}")
                raise Exception(error_redirect)
        # Get redirect URL
        redirect_url = response.headers["Location"].replace("&nfy=1", "")
        request_id = redirect_url.split("id=")[-1]
        self.session.get(f"{BING_URL}{redirect_url}")
        # https://www.bing.com/images/create/async/results/{ID}?q={PROMPT}
        polling_url = f"{BING_URL}/images/create/async/results/{request_id}?q={url_encoded_prompt}"
        # Poll for results
        if self.debug_file:
            self.debug("Polling and waiting for result")
        if not self.quiet:
            print("Waiting for results...")
        start_wait = time.time()
        while True:
            if int(time.time() - start_wait) > 200:
                if self.debug_file:
                    self.debug(f"ERROR: {error_timeout}")
                raise Exception(error_timeout)
            if not self.quiet:
                print(".", end="", flush=True)
            response = self.session.get(polling_url)
            if response.status_code != 200:
                if self.debug_file:
                    self.debug(f"ERROR: {error_noresults}")
                raise Exception(error_noresults)
            if not response.text or response.text.find("errorMessage") != -1:
                time.sleep(1)
                continue
            else:
                break
        # Use regex to search for src=""
        image_links = regex.findall(r'src="([^"]+)"', response.text)
        # Remove size limit
        normal_image_links = [link.split("?w=")[0] for link in image_links]
        # Remove duplicates
        normal_image_links = list(set(normal_image_links))

        # Bad images
        bad_images = [
            "https://r.bing.com/rp/in-2zU3AJUdkgFe7ZKv19yPBHVs.png",
            "https://r.bing.com/rp/TX9QuO3WzcCJz1uaaSwQAz39Kb0.jpg",
        ]
        for img in normal_image_links:
            if img in bad_images:
                raise Exception("Bad images")
        # No images
        if not normal_image_links:
            raise Exception(error_no_images)

        normal_image_links.remove("https://r.bing.com/rp/gmZtdJVd-klWl3XWpa6-ni1FU3M.svg")
        return normal_image_links

    def save_images(
            self,
            links: list,
            output_dir: str,
            file_name: str = None,
            download_count: int = None,
    ) -> None:
        """
        Saves images to output directory
        Parameters:
            links: list[str]
            output_dir: str
            file_name: str
            download_count: int
        """
        if self.debug_file:
            self.debug(download_message)
        if not self.quiet:
            print(download_message)
        with contextlib.suppress(FileExistsError):
            os.mkdir(output_dir)
        try:
            fn = f"{file_name}-" if file_name else ""
            jpeg_index = 0

            if download_count:
                links = links[:download_count]

            for link in links:
                while os.path.exists(
                        os.path.join(output_dir, f"{fn}{jpeg_index}.jpeg")
                ):
                    jpeg_index += 1
                for i in range(0, 100):
                    while True:
                        try:
                            response = self.session.get(link)
                        except Exception:
                            continue
                        break
                if response.status_code != 200 or sys.getsizeof(response.content) < 5000:
                    raise Exception("Could not download image: " + link)
                # save response to file
                with open(
                        os.path.join(output_dir, f"{fn}{jpeg_index}.jpeg"), "wb"
                ) as output_file:
                    output_file.write(response.content)
                jpeg_index += 1

        except requests.exceptions.MissingSchema as url_exception:
            raise Exception(
                "Inappropriate contents found in the generated images. Please try again or try another prompt.",
            ) from url_exception


class ImageGenAsync:
    """
    Image generation by Microsoft Bing
    Parameters:
        auth_cookie: str
    Optional Parameters:
        debug_file: str
        quiet: bool
        all_cookies: list[dict]
    """

    def __init__(
            self,
            auth_cookie: str = None,
            debug_file: Union[str, None] = None,
            quiet: bool = False,
            all_cookies: List[Dict] = None,
    ) -> None:
        if auth_cookie is None and not all_cookies:
            raise Exception("No auth cookie provided")
        self.session = httpx.AsyncClient(
            headers=HEADERS,
            trust_env=True,
        )
        if auth_cookie:
            self.session.cookies.update({"_U": auth_cookie})
        if all_cookies:
            for cookie in all_cookies:
                self.session.cookies.update(
                    {cookie["name"]: cookie["value"]},
                )
        self.quiet = quiet
        self.debug_file = debug_file
        if self.debug_file:
            self.debug = partial(debug, self.debug_file)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *excinfo) -> None:
        await self.session.aclose()

    async def get_images(self, prompt: str) -> list:
        """
        Fetches image links from Bing
        Parameters:
            prompt: str
        """
        if not self.quiet:
            print("Sending request...")
        url_encoded_prompt = requests.utils.quote(prompt)
        # https://www.bing.com/images/create?q=<PROMPT>&rt=3&FORM=GENCRE
        url = f"{BING_URL}/images/create?q={url_encoded_prompt}&rt=3&FORM=GENCRE"
        payload = f"q={url_encoded_prompt}&qs=ds"
        response = await self.session.post(
            url,
            follow_redirects=False,
            data=payload,
        )
        content = response.text
        if "this prompt has been blocked" in content.lower():
            raise Exception(
                "Your prompt has been blocked by Bing. Try to change any bad words and try again.",
            )
        if response.status_code != 302:
            # if rt4 fails, try rt3
            url = f"{BING_URL}/images/create?q={url_encoded_prompt}&rt=4&FORM=GENCRE"
            response = await self.session.post(
                url,
                follow_redirects=False,
                timeout=200,
            )
            if response.status_code != 302:
                print(f"ERROR: {response.text}")
                raise Exception("Redirect failed")
        # Get redirect URL
        redirect_url = response.headers["Location"].replace("&nfy=1", "")
        request_id = redirect_url.split("id=")[-1]
        await self.session.get(f"{BING_URL}{redirect_url}")
        # https://www.bing.com/images/create/async/results/{ID}?q={PROMPT}
        polling_url = f"{BING_URL}/images/create/async/results/{request_id}?q={url_encoded_prompt}"
        # Poll for results
        if not self.quiet:
            print("Waiting for results...")
        while True:
            if not self.quiet:
                print(".", end="", flush=True)
            # By default, timeout is 300s, change as needed
            response = await self.session.get(polling_url)
            if response.status_code != 200:
                raise Exception("Could not get results")
            content = response.text
            if content and content.find("errorMessage") == -1:
                break

            await asyncio.sleep(1)
            continue
        # Use regex to search for src=""
        image_links = regex.findall(r'src="([^"]+)"', content)
        # Remove size limit
        normal_image_links = [link.split("?w=")[0] for link in image_links]
        # Remove duplicates
        normal_image_links = list(set(normal_image_links))

        # Bad images
        bad_images = [
            "https://r.bing.com/rp/in-2zU3AJUdkgFe7ZKv19yPBHVs.png",
            "https://r.bing.com/rp/TX9QuO3WzcCJz1uaaSwQAz39Kb0.jpg",
        ]
        for im in normal_image_links:
            if im in bad_images:
                raise Exception("Bad images")
        # No images
        if not normal_image_links:
            raise Exception("No images")
        return normal_image_links

    async def save_images(
            self,
            links: list,
            output_dir: str,
            download_count: int,
            file_name: str = None,
    ) -> None:
        """
        Saves images to output directory
        """

        if self.debug_file:
            self.debug(download_message)
        if not self.quiet:
            print(download_message)
        with contextlib.suppress(FileExistsError):
            os.mkdir(output_dir)
        try:
            fn = f"{file_name}_" if file_name else ""
            jpeg_index = 0

            for link in links[:download_count]:
                while os.path.exists(
                        os.path.join(output_dir, f"{fn}{jpeg_index}.jpeg")
                ):
                    jpeg_index += 1
                response = await self.session.get(link)
                if response.status_code != 200:
                    raise Exception("Could not download image")
                # save response to file
                with open(
                        os.path.join(output_dir, f"{fn}{jpeg_index}.jpeg"), "wb"
                ) as output_file:
                    output_file.write(response.content)
                jpeg_index += 1
        except httpx.InvalidURL as url_exception:
            raise Exception(
                "Inappropriate contents found in the generated images. Please try again or try another prompt.",
            ) from url_exception


async def async_image_gen(
        prompt: str,
        download_count: int,
        output_dir: str,
        u_cookie=None,
        debug_file=None,
        quiet=False,
        all_cookies=None,
):
    async with ImageGenAsync(
            u_cookie,
            debug_file=debug_file,
            quiet=quiet,
            all_cookies=all_cookies,
    ) as image_generator:
        images = await image_generator.get_images(prompt)
        await image_generator.save_images(
            images, output_dir=output_dir, download_count=download_count
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-U", help="Auth cookie from browser", type=str)
    parser.add_argument("--cookie-file", help="File containing auth cookie", type=str)
    parser.add_argument(
        "--prompt",
        help="Prompt to generate images for",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        help="Output directory",
        type=str,
        default="./output",
    )

    parser.add_argument(
        "--download-count",
        help="Number of images to download, value must be less than five",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--debug-file",
        help="Path to the file where debug information will be written.",
        type=str,
    )

    parser.add_argument(
        "--quiet",
        help="Disable pipeline messages",
        action="store_true",
    )
    parser.add_argument(
        "--asyncio",
        help="Run ImageGen using asyncio",
        action="store_true",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print the version number",
    )

    args = parser.parse_args()

    if args.version:
        print(pkg_resources.get_distribution("BingImageCreator").version)
        sys.exit()

    # Load auth cookie
    cookie_json = None
    if args.cookie_file is not None:
        with contextlib.suppress(Exception):
            with open(args.cookie_file, encoding="utf-8") as file:
                cookie_json = json.load(file)

    if args.U is None and args.cookie_file is None:
        raise Exception("Could not find auth cookie")

    if args.download_count > 4:
        raise Exception("The number of downloads must be less than five")

    if not args.asyncio:
        prompt_list = ["Tattoo with a Rose, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Rose, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Rose, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Rose, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Rose, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Rose, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Rose, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Rose, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Rose, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Rose, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Flower, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Flower, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Flower, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Flower, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Flower, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Flower, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Flower, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Flower, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Flower, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Flower, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Moon, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Moon, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Moon, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Moon, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Moon, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Moon, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Moon, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Moon, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Moon, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Moon, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lotus flowers, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lotus flowers, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lotus flowers, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lotus flowers, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lotus flowers, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lotus flowers, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lotus flowers, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lotus flowers, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lotus flowers, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lotus flowers, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cherry blossom, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cherry blossom, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cherry blossom, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cherry blossom, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cherry blossom, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cherry blossom, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cherry blossom, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cherry blossom, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cherry blossom, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cherry blossom, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mountain, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mountain, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mountain, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mountain, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mountain, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mountain, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mountain, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mountain, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mountain, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mountain, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sun , Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sun , Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sun , Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sun , Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sun , Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sun , Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sun , Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sun , Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sun , Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sun , Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Fire, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Fire, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Fire, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Fire, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Fire, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Fire, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Fire, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Fire, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Fire, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Fire, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Viking, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Viking, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Viking, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Viking, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Viking, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Viking, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Viking, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Viking, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Viking, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Viking, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Medusa, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Medusa, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Medusa, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Medusa, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Medusa, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Medusa, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Medusa, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Medusa, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Medusa, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Medusa, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Samurai, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Samurai, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Samurai, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Samurai, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Samurai, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Samurai, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Samurai, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Samurai, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Samurai, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Samurai, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anubis, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anubis, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anubis, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anubis, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anubis, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anubis, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anubis, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anubis, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anubis, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anubis, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Unalome, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Unalome, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Unalome, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Unalome, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Unalome, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Unalome, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Unalome, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Unalome, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Unalome, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Unalome, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Tree of LIfe, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Tree of LIfe, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Tree of LIfe, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Tree of LIfe, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Tree of LIfe, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Tree of LIfe, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Tree of LIfe, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Tree of LIfe, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Tree of LIfe, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Tree of LIfe, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memento Mori, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memento Mori, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memento Mori, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memento Mori, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memento Mori, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memento Mori, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memento Mori, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memento Mori, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memento Mori, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memento Mori, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Skull, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Skull, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Skull, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Skull, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Skull, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Skull, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Skull, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Skull, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Skull, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Skull, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Semicolon, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Semicolon, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Semicolon, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Semicolon, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Semicolon, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Semicolon, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Semicolon, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Semicolon, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Semicolon, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Semicolon, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mandala, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mandala, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mandala, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mandala, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mandala, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mandala, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mandala, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mandala, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mandala, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Mandala, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Compass, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Compass, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Compass, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Compass, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Compass, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Compass, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Compass, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Compass, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Compass, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Compass, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cross, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cross, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cross, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cross, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cross, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cross, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cross, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cross, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cross, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Cross, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Zodiac Signs, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Zodiac Signs, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Zodiac Signs, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Zodiac Signs, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Zodiac Signs, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Zodiac Signs, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Zodiac Signs, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Zodiac Signs, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Zodiac Signs, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Zodiac Signs, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anchor, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anchor, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anchor, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anchor, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anchor, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anchor, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anchor, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anchor, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anchor, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Anchor, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Heart, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Heart, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Heart, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Heart, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Heart, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Heart, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Heart, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Heart, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Heart, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Heart, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Music, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Music, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Music, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Music, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Music, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Music, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Music, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Music, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Music, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Music, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Hourglass, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Hourglass, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Hourglass, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Hourglass, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Hourglass, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Hourglass, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Hourglass, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Hourglass, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Hourglass, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Hourglass, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Friendship, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Friendship, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Friendship, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Friendship, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Friendship, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Friendship, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Friendship, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Friendship, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Friendship, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Friendship, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memorial, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memorial, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memorial, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memorial, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memorial, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memorial, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memorial, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memorial, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memorial, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Memorial, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Infinity, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Infinity, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Infinity, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Infinity, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Infinity, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Infinity, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Infinity, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Infinity, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Infinity, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Infinity, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Yin Yang, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Yin Yang, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Yin Yang, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Yin Yang, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Yin Yang, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Yin Yang, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Yin Yang, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Yin Yang, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Yin Yang, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Yin Yang, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wings, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wings, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wings, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wings, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wings, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wings, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wings, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wings, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wings, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wings, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Clock, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Clock, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Clock, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Clock, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Clock, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Clock, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Clock, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Clock, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Clock, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Clock, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wave, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wave, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wave, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wave, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wave, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wave, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wave, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wave, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wave, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Wave, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Diamond, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Diamond, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Diamond, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Diamond, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Diamond, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Diamond, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Diamond, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Diamond, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Diamond, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Diamond, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eyeball, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eyeball, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eyeball, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eyeball, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eyeball, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eyeball, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eyeball, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eyeball, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eyeball, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eyeball, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sword, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sword, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sword, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sword, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sword, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sword, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sword, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sword, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sword, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Sword, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Feather, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Feather, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Feather, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Feather, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Feather, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Feather, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Feather, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Feather, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Feather, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Feather, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Dream Catcher, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Dream Catcher, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Dream Catcher, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Dream Catcher, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Dream Catcher, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Dream Catcher, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Dream Catcher, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Dream Catcher, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Dream Catcher, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Dream Catcher, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Arrow, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Arrow, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Arrow, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Arrow, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Arrow, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Arrow, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Arrow, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Arrow, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Arrow, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Arrow, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eeyore, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eeyore, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eeyore, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eeyore, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eeyore, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eeyore, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eeyore, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eeyore, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eeyore, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Eeyore, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a One Piece, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a One Piece, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a One Piece, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a One Piece, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a One Piece, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a One Piece, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a One Piece, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a One Piece, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a One Piece, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a One Piece, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Naruto, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Naruto, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Naruto, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Naruto, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Naruto, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Naruto, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Naruto, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Naruto, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Naruto, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Naruto, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star Wars, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star Wars, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star Wars, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star Wars, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star Wars, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star Wars, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star Wars, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star Wars, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star Wars, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Star Wars, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Harry Potter, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Harry Potter, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Harry Potter, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Harry Potter, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Harry Potter, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Harry Potter, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Harry Potter, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Harry Potter, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Harry Potter, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Harry Potter, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a John Wick, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a John Wick, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a John Wick, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a John Wick, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a John Wick, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a John Wick, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a John Wick, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a John Wick, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a John Wick, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a John Wick, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lord of the Rings, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lord of the Rings, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lord of the Rings, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lord of the Rings, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lord of the Rings, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lord of the Rings, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lord of the Rings, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lord of the Rings, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lord of the Rings, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Lord of the Rings, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Avengers, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Avengers, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Avengers, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Avengers, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Avengers, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Avengers, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Avengers, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Avengers, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Avengers, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Avengers, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Pokemon, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Pokemon, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Pokemon, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Pokemon, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Pokemon, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Pokemon, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Pokemon, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Pokemon, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Pokemon, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Pokemon, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Batman, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Batman, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Batman, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Batman, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Batman, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Batman, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Batman, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Batman, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Batman, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Batman, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Superman, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Superman, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Superman, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Superman, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Superman, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Superman, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Superman, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Superman, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Superman, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Superman, Dotwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Joker, Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Joker, Realism Style, High Quality, On Body, Small Size",
                       "Tattoo with a Joker, Neo-Traditional Style, High Quality, On Body, Small Size",
                       "Tattoo with a Joker, Watercolor Style, High Quality, On Body, Small Size",
                       "Tattoo with a Joker, Blackwork Style, High Quality, On Body, Small Size",
                       "Tattoo with a Joker, Tribal Style, High Quality, On Body, Small Size",
                       "Tattoo with a Joker, Japanese Style, High Quality, On Body, Small Size",
                       "Tattoo with a Joker, Geometric Style, High Quality, On Body, Small Size",
                       "Tattoo with a Joker, Trash Polka Style, High Quality, On Body, Small Size",
                       "Tattoo with a Joker, Dotwork Style, High Quality, On Body, Small Size"]

        image_generator = ImageGen(
            args.U,
            args.debug_file,
            args.quiet,
            all_cookies=cookie_json,
        )

#        random.shuffle(prompt_list)
        for prompt in prompt_list:
            time.sleep(random.uniform(1.5, 7))
            print("start: " + prompt)
            try:
                image_generator.save_images(
                    image_generator.get_images(prompt),
                    output_dir=args.output_dir,
                    download_count=args.download_count,
                    file_name=slugify(prompt),
                )
            except Exception as e:
                print("error: " + prompt)
    else:
        asyncio.run(
            async_image_gen(
                args.prompt,
                args.download_count,
                args.output_dir,
                args.U,
                args.debug_file,
                args.quiet,
                all_cookies=cookie_json,
            ),
        )


if __name__ == "__main__":
    main()
