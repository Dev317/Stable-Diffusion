export async function POST(request: Request) {
	const data = await request.formData();
	console.log(data);
	return new Response(JSON.stringify({
		image: 'http://localhost:3000/dog.jpg',
		category: "Dog bark"
	}));
}
