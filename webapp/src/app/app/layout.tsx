"use client";
import {FilePondFile} from "filepond";
import {Dispatch, createContext, useContext, useState} from "react";

interface FilesContextProps {
	files: FilePondFile["file"][];
	setFiles: Dispatch<FilePondFile["file"][]>;
}
const FilesContext = createContext<FilesContextProps>({
	files: [],
	setFiles: () => {},
});
export const useFilesContext = () => useContext(FilesContext);

export default function AppLayout({children}: {children: React.ReactNode}) {
	const [files, setFiles] = useState<FilePondFile["file"][]>([]);
	return (
		<FilesContext.Provider value={{files, setFiles}}>
			{children}
		</FilesContext.Provider>
	);
}
