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
import {FilePondFile} from "filepond";
import {useRouter} from "next/navigation";

const inter = Inter({subsets: ["latin"]});

export default function FileForm() {
	const [show, setShow] = useState(false);
	useEffect(function revealOnMount() {
		setShow(true);
	}, []);

	const router = useRouter();
	const [isPending, startTransition] = useTransition();
	const [isFetching, setIsFetching] = useState(false);
	const isMutating = isPending || isFetching;

	const [files, setFiles] = useState<FilePondFile["file"][]>([]);
	const handleSubmit = useCallback(
		async (ev: FormEvent<HTMLFormElement>) => {
			ev.preventDefault();
			try {
				setIsFetching(true);
				const data = new FormData();
				data.append("wav_file", files[0]);
				const response = await fetch(`/api/audio2image`, {
					method: "POST",
					body: data,
				});
				const responseData = await response.json();
				const encodedImageURI = encodeURIComponent(responseData.image);
				startTransition(() => {
					router.push(
						`/?image=${encodedImageURI}&category=${responseData.category}`,
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
		<form
			onSubmit={handleSubmit}
			method="POST"
			action="/api/audio2image"
			className={twMerge(
				"flex flex-col items-center gap-4 opacity-0 transition-opacity duration-1000",
				show && "opacity-100",
				isMutating && "animate-pulse",
			)}
		>
			<FileUpload required files={files} onFilesChange={setFiles} />
			<button
				type="submit"
				className={`${inter.className} text-white bg-blue-700 px-4 py-2 aspect-square rounded-full hover:bg-blue-600 hover:shadow hover:shadow-blue-700/50 transition-all duration-500`}
			>
				-&gt;
			</button>
		</form>
	);
}