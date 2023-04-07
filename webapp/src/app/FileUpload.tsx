"use client";
import {Inter} from "next/font/google";
import {Dispatch, useCallback} from "react";
import {FilePondFile} from "filepond";
import {FilePond, registerPlugin} from "react-filepond";
import FilePondPluginMediaPreview from "filepond-plugin-media-preview";

import "filepond/dist/filepond.min.css";
import "filepond-plugin-media-preview/dist/filepond-plugin-media-preview.min.css";
import "./FileUpload.css";

const inter = Inter({subsets: ["latin"]});

registerPlugin(FilePondPluginMediaPreview);

export interface FileUploadProps {
	required?: boolean;
	files?: FilePondFile["file"][];
	onFilesChange?: Dispatch<FilePondFile["file"][]>;
}

export default function FileUpload({
	required,
	files,
	onFilesChange,
}: FileUploadProps) {
	const handleFilesUpdate = useCallback(
		(files: FilePondFile[]) => {
			onFilesChange?.(files.map((f) => f.file));
		},
		[onFilesChange],
	);

	return (
		<div className={`${inter.className} w-96 min-h-20`}>
			<FilePond
				name="wav_file"
				files={files}
				onupdatefiles={handleFilesUpdate}
				required={required}
			/>
		</div>
	);
}
