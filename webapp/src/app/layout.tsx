import "./globals.css";

export const metadata = {
	title: "Audio Thumbnail Generator",
	description: "Get thumbnails quickly for your audio clips",
};

export default function RootLayout({children}: {children: React.ReactNode}) {
	return (
		<html lang="en">
			<body>{children}</body>
		</html>
	);
}
