import styles from "./Loader.module.css";

export default function Loader() {
	return (
		<span className="flex gap-2 h-12">
			{Array.from({length: 5}).map((_, i) => (
				<span
					key={i}
					className={`w-1 bg-foreground rounded-full ${styles.loader}`}
					style={
						{
							"--delay": `${i * 100}ms`,
						} as any
					}
				/>
			))}
		</span>
	);
}
