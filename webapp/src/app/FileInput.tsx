"use client";
import {useCallback, useEffect, useState} from "react";
import {FilePondFile} from "filepond";
import {FilePond, registerPlugin} from "react-filepond";
import FilePondPluginMediaPreview from "filepond-plugin-media-preview";
import {twMerge} from "tailwind-merge";

import "filepond/dist/filepond.min.css";
import "filepond-plugin-media-preview/dist/filepond-plugin-media-preview.min.css";
import "./FileInput.css";

registerPlugin(FilePondPluginMediaPreview);

export default function FileInput() {
	const [files, setFiles] = useState<FilePondFile["file"][]>([]);
	const handleFilesUpdate = useCallback((files: FilePondFile[]) => {
		setFiles(files.map((f) => f.file));
	}, []);

	const [show, setShow] = useState(false);
	useEffect(function revealOnMount() {
		setShow(true);
	}, []);

	return (
		<div
			className={twMerge(
				"w-96 opacity-0 transition-opacity duration-1000",
				show && "opacity-100",
			)}
		>
			<FilePond files={files} onupdatefiles={handleFilesUpdate} />
		</div>
	);
}
