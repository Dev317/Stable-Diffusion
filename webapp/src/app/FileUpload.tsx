"use client";
import {useCallback, useState} from "react";
import {FilePondFile} from "filepond";
import {FilePond, registerPlugin} from "react-filepond";
import FilePondPluginMediaPreview from "filepond-plugin-media-preview";

import "filepond/dist/filepond.min.css";
import "filepond-plugin-media-preview/dist/filepond-plugin-media-preview.min.css";
import "./FileUpload.css";

registerPlugin(FilePondPluginMediaPreview);

export interface FileUploadProps {
	required?: boolean;
}

export default function FileUpload({required}: FileUploadProps) {
	const [files, setFiles] = useState<FilePondFile["file"][]>([]);
	const handleFilesUpdate = useCallback((files: FilePondFile[]) => {
		setFiles(files.map((f) => f.file));
	}, []);

	return (
		<div className="w-96 min-h-20">
			<FilePond
				name="file"
				files={files}
				onupdatefiles={handleFilesUpdate}
				required={required}
			/>
		</div>
	);
}
