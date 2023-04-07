import Image from "next/image";
import {Inter} from "next/font/google";
import FileForm from "./FileForm";

const inter = Inter({subsets: ["latin"]});

interface PageParams {
	searchParams: {
		image?: string;
		category?: string;
	};
}

export default function Home({searchParams}: PageParams) {
	return (
		<main className="flex min-h-screen flex-col items-center justify-between p-24">
			<div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
				<p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
					Audio Thumbnail Generator
				</p>
				<div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify-center bg-gradient-to-t from-white via-white dark:from-black dark:via-black lg:static lg:h-auto lg:w-auto lg:bg-none">
					<a
						className="pointer-events-none flex place-items-center gap-2 p-8 lg:pointer-events-auto lg:p-0"
						href="https://vercel.com?utm_source=create-next-app&utm_medium=appdir-template&utm_campaign=create-next-app"
						target="_blank"
						rel="noopener noreferrer"
					>
						By Troll Class Team 1
					</a>
				</div>
			</div>

			<div className="relative flex place-items-center before:absolute before:h-[300px] before:w-[480px] before:-translate-x-1/2 before:rounded-full before:bg-gradient-radial before:from-white before:to-transparent before:blur-2xl before:content-[''] before:pointer-events-none after:pointer-events-none after:absolute before:-z-10 after:-z-20 after:h-[180px] after:w-[240px] after:translate-x-1/3 after:bg-gradient-conic after:from-sky-200 after:via-blue-200 after:blur-2xl after:content-[''] before:dark:bg-gradient-to-br before:dark:from-transparent before:dark:to-blue-700 before:dark:opacity-10 after:dark:from-sky-900 after:dark:via-[#0141ff] after:dark:opacity-40 before:lg:h-[360px]">
				{searchParams.image ? (
					<div className="relative flex flex-col gap-4 items-center">
						<Image
							src={searchParams.image}
							alt="The generated image"
							width={480}
							height={480}
							className="rounded-xl border border-gray-300 dark:border-neutral-800"
						/>
						<p className="flex justify-center border-b border-gray-300 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit w-auto rounded-xl border bg-gray-200/50 p-4">
							{searchParams.category}
						</p>
						<Image
							src={searchParams.image}
							alt="The generated image"
							width={480}
							height={480}
							className="absolute inset-0 blur-3xl -z-20 opacity-50 scale-125"
						/>
					</div>
				) : (
					<FileForm />
				)}
			</div>

			<footer className="w-full max-w-5xl mb-32 grid text-center lg:mb-0 lg:grid-cols-3 lg:text-left">
				<a
					href="/about"
					className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30 flex flex-col items-center lg:items-start"
				>
					<h2 className={`${inter.className} mb-3 text-2xl font-semibold`}>
						Architecture{" "}
						<span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
							-&gt;
						</span>
					</h2>
					<p
						className={`${inter.className} m-0 max-w-[30ch] text-sm opacity-50`}
					>
						Learn more about how we built this.
					</p>
				</a>
			</footer>
		</main>
	);
}
