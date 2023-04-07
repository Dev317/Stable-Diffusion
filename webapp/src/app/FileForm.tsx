"use client";
import {Inter} from "next/font/google";
import FileUpload from "./FileUpload";
import {useEffect, useState} from "react";
import {twMerge} from "tailwind-merge";

const inter = Inter({subsets: ["latin"]});

export default function FileForm() {
	const [show, setShow] = useState(false);
	useEffect(function revealOnMount() {
		setShow(true);
	}, []);

	return (
		<form
			method="POST"
			action="/api/audio2image"
			className={twMerge(
				"flex flex-col items-center gap-4 opacity-0 transition-opacity duration-1000",
				show && "opacity-100",
			)}
		>
			<FileUpload required />
			<button
				type="submit"
				className={`${inter.className} text-white bg-blue-700 px-4 py-2 aspect-square rounded-full hover:bg-blue-600 hover:shadow hover:shadow-blue-700/50 transition-all duration-500`}
			>
				-&gt;
			</button>
		</form>
	);
}
