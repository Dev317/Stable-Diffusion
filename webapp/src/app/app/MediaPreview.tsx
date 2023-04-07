"use client";
import Image from "next/image";
import FileUpload from "./FileUpload";
import {useFilesContext} from "./layout";

export interface MediaPreviewProps {
	image: string;
	category?: string;
}

export default function MediaPreview({image, category}: MediaPreviewProps) {
	const {files} = useFilesContext();
	return (
		<div className="flex flex-col gap-4 items-center">
			<p className="p-3 bg-gray-200/30 dark:bg-zinc-800/30 rounded-xl border border-gray-300 dark:border-neutral-800 backdrop-blur-2xl text-sm font-medium">
				{category}
			</p>
			<div className="relative">
				<Image
					src={image}
					alt="The generated image"
					width={480}
					height={480}
					className="rounded-xl border border-gray-300 dark:border-neutral-800"
				/>
				<Image
					src={image}
					alt="The generated image"
					width={480}
					height={480}
					className="absolute inset-0 blur-3xl -z-20 opacity-50 scale-125"
				/>
			</div>
			<div className="flex gap-4 w-full">
				<FileUpload files={files} />
			</div>
		</div>
	);
}
