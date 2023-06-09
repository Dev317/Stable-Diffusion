"use client";
import {Inter} from "next/font/google";
import FileUpload from "./FileUpload";
import {
	FormEvent,
	useCallback,
	useEffect,
	useState,
	useTransition,
} from "react";
import {twMerge} from "tailwind-merge";
import {useRouter} from "next/navigation";
import Loader from "./Loader";
import {useFilesContext} from "./layout";
import {FilePondFile} from "filepond";

const inter = Inter({subsets: ["latin"]});

async function fetchResponse(file: FilePondFile["file"]) {
	if (process.env.NEXT_PUBLIC_MOCK_API === "TRUE") {
		await new Promise((res) => {
			setTimeout(res, 6000);
		});
		return {
			image: "/dog.jpg",
			category: "dog bark",
			image_title: "dog",
			width: 512,
			height: 512,
			probability: 0.85,
		};
	}

	const data = new FormData();
	data.append("wav_file", file);
	const response = await fetch(
		`${process.env.NEXT_PUBLIC_AUDIO_TO_IMAGE_API_HOST}/api/audio2image`,
		{
			method: "POST",
			body: data,
		},
	);
	return await response.json();
}

export default function FileForm() {
	const [show, setShow] = useState(false);
	useEffect(function revealOnMount() {
		setShow(true);
	}, []);

	const router = useRouter();
	const [isPending, startTransition] = useTransition();
	const [isFetching, setIsFetching] = useState(false);
	const isMutating = isPending || isFetching;

	const {files, setFiles} = useFilesContext();

	const handleSubmit = useCallback(
		async (ev: FormEvent<HTMLFormElement>) => {
			ev.preventDefault();
			try {
				setIsFetching(true);
				const responseData = await fetchResponse(files[0]);
				console.log(responseData);
				const encodedImageURI = encodeURIComponent(responseData.image);
				const {category, probability} = responseData;
				startTransition(() => {
					router.push(
						`app/?image=${encodedImageURI}&category=${category}&probability=${probability}`,
					);
				});
			} catch (err) {
				console.error(err);
			} finally {
				setIsFetching(false);
			}
		},
		[router, files],
	);

	return (
		<div className="relative">
			<form
				onSubmit={handleSubmit}
				method="POST"
				action="/api/audio2image"
				className={twMerge(
					"relative flex items-center gap-4 opacity-0 transition-opacity duration-1000",
					show && "opacity-100",
					isMutating && "opacity-50 animate-pulse",
				)}
			>
				<FileUpload required files={files} onFilesChange={setFiles} />
				{files.length > 0 && (
					<button
						type="submit"
						className={`${inter.className} text-white bg-blue-700 aspect-square rounded-full hover:bg-blue-600 hover:shadow hover:shadow-blue-700/50 transition-all duration-500 w-16 h-16 flex items-center justify-center`}
					>
						-&gt;
					</button>
				)}
			</form>
			<div
				className={twMerge(
					"absolute inset-0 flex justify-center items-center opacity-0 transition-opacity duration-1000 pointer-events-none",
					isMutating && "opacity-100",
				)}
			>
				<Loader />
			</div>
		</div>
	);
}
